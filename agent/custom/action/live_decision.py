"""Fail-closed Maa bridge for the DMM Live decision preview."""

from __future__ import annotations

import os
import re
import sys
import json
import math
import time
import ctypes
import inspect
from types import SimpleNamespace
from typing import Any, NoReturn
from pathlib import Path
from dataclasses import replace

from utils import logger
from maa.context import Context
from live_decision import (
    UNKNOWN,
    PROTOCOL_VERSION,
    DmmTurnTracker,
    LiveSafetyConfig,
    LiveSelectionError,
    LiveCardLayoutError,
    LiveUpgradeConsensus,
    WeightedPolicyProvider,
    SkillCardFeatureCatalog,
    DmmLiveVisualStateProvider,
    DeterministicLiveSafetyFilter,
    locate_dmm_self_rank,
    live_select_label_box,
    build_live_title_views,
    classify_live_card_views,
    normalise_live_candidates,
    remaining_live_card_clicks,
    resolve_live_title_consensus,
    resolve_live_selected_slot_consensus,
)
from card_selection import EmbeddingCardRecognizer
from maa.custom_action import CustomAction
from route.battle_policy import FORMAL_PROFILES
from card_selection.model import frame_identifier, isolate_card_candidates
from live_decision.layout import calibrated_live_client_size
from maa.agent.agent_server import AgentServer
from live_decision.selection import (
    same_live_candidate_geometry as _same_live_candidate_geometry,
)
from live_decision.selection import (
    live_selection_geometry_evidence as _live_selection_geometry_evidence,
)
from live_decision.hif_visual import (
    HifLiveRunContext,
    HifFrameTitleCache,
    hif_select_label_rois,
    hif_selection_click_box,
    hif_selected_row_proposal,
    locate_hif_opening_self_rank,
    hif_unselected_row_candidates,
    hif_selected_fragment_proposal,
    locate_hif_selected_detail_title,
)
from live_decision.masked_art import FrozenMaskedArt
from live_decision.hif_capture_adapter import captured_at as hif_captured_at

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_runtime_path(source_relative: Path, installed_relative: Path) -> Path:
    """Resolve repository and packaged layouts without mutating either tree."""

    candidates = (PROJECT_ROOT / source_relative, PROJECT_ROOT / installed_relative)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


MODEL_ROOT = _resolve_runtime_path(
    Path("assets/resource/base/model/classify/card_embedding"),
    Path("resource/base/model/classify/card_embedding"),
)
POLICY_PATH = _resolve_runtime_path(
    Path("assets/data/live_decision_presets.yaml"),
    Path("data/live_decision_presets.yaml"),
)
CATALOG_PATH = _resolve_runtime_path(
    Path("assets/data/skill_cards.csv"),
    Path("data/skill_cards.csv"),
)
DEBUG_TARGET_HWND_ENV = "MAA_GAKUMAS_DEBUG_TARGET_HWND"
DEBUG_TARGET_PID_ENV = "MAA_GAKUMAS_DEBUG_TARGET_PID"
DEBUG_TARGET_WINDOW_CLASS = "UnityWndClass"
DEBUG_TARGET_WINDOW_TITLE = "gakumas"
DEBUG_FOREGROUND_PREPARATION_STEP_SECONDS = 0.25
LIVE_CARD_ART_BOXES = (
    (53, 899, 180, 180),
    (265, 919, 190, 160),
    (497, 919, 170, 160),
)

LIVE_LAYOUT_V1 = {
    "turns_remaining": ([35, 40, 75, 90], r"^\d{1,2}$"),
    # The heart value is stamina.  Keep its OCR crop left of the shield/cost
    # column so the shield, selected-card preview, and P-item counter cannot be
    # promoted into card-cost safety.
    "stamina": ([555, 185, 93, 65], r"^\d{1,3}$"),
    "score_multiplier": ([100, 55, 145, 95], r"^\d{1,4}%$"),
}
LIVE_SHIELD_ROI = [648, 155, 64, 45]
LIVE_SHIELD_PREVIEW_DELTA_ROI = [648, 200, 64, 35]
LIVE_STAMINA_PREVIEW_DELTA_ROI = [568, 250, 64, 35]
LIVE_SELECTED_EFFECTIVE_COST_RELATIVE_ROI = (140, 174, 50, 45)
LIVE_P_ITEM_USE_COUNT_REGION = [688, 235, 32, 45]
LIVE_RESOURCE_DELTA_RED_MIN_PIXELS = 40
LIVE_RESOURCE_DELTA_RED_MAX_PIXELS = 700
LIVE_RESOURCE_DELTA_RED_MIN_WIDTH = 10
LIVE_RESOURCE_DELTA_RED_MIN_HEIGHT = 8
LIVE_RESOURCE_DELTA_RED_COMPONENT_MIN_AREA = 3
LIVE_SELF_SCORE_ZERO_ROI_OFFSET = (-75, 120, 40, 45)
LIVE_SELF_SCORE_ZERO_MIN_CONFIDENCE = 0.90
LIVE_SELF_SCORE_ZERO_BOX_WIDTH_RANGE = (20, 36)
LIVE_SELF_SCORE_ZERO_BOX_HEIGHT_RANGE = (28, 42)
LIVE_AUTHORITATIVE_OCR_MOSAIC_CELL = (240, 100)
LIVE_AUTHORITATIVE_OCR_MOSAIC_SHAPE = (400, 720)
LIVE_AUTHORITATIVE_OCR_MOSAIC_ABSOLUTE_MIN_CONFIDENCE = 0.95
LIVE_AUTHORITATIVE_DELTA_MOSAIC_CELL = (168, 110)
LIVE_AUTHORITATIVE_DELTA_MOSAIC_SHAPE = (110, 336)
LIVE_FIELD_LIMITS = {
    "turns_remaining": (1, 12),
    "stamina": (0, 100),
    "shield": (0, 100),
    "score": (0, 999999999),
    "score_multiplier": (0.01, 99.99),
}
TURN_TRACKER = DmmTurnTracker()
SELECTION_SETTLE_SECONDS = 0.35
SELECTION_FRAME_INTERVAL_SECONDS = 0.08
LIVE_MIDDLE_OCR_INLINE_POPUP_RESERVE_MS = 80.0
LIVE_SELECTED_TRIPLE_CAPTURE_RESERVE_MS = 160.0
EXECUTION_POST_SETTLE_SECONDS = 0.35
EXECUTION_POST_CONFIRM_ATTEMPTS = 6
LIVE_SOFT_UNKNOWN_FIELDS = frozenset(
    {"items", "field_effects", "statuses", "max_stamina"}
)
LIVE_POST_TERMINAL_NODES = (
    "ProduceFailedFlag",
    "ProduceNIAFailedFlag",
    "ProduceFinished",
)


def _parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Live decision parameters must be a JSON object")
    return parsed


def _strict_finite_number(
    params: dict[str, Any],
    name: str,
    *,
    default: float | None,
    lower: float,
    upper: float,
) -> float | None:
    raw = params.get(name, default)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{name} must be a finite JSON number")
    value = float(raw)
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} is outside the approved range")
    return value


def _validate_runtime_params(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate all input-capable settings before any Live-card observation click."""

    params = dict(raw)
    explicit_identity_thresholds = all(
        name in raw and raw[name] is not None
        for name in ("debug_identity_threshold", "debug_identity_min_margin")
    )
    if "debug_selected_cost_min_confidence" in raw:
        raise ValueError("debug_selected_cost_min_confidence is obsolete")
    if "debug_expected_hand_count" in raw:
        raise ValueError("debug_expected_hand_count is obsolete")
    transaction_threshold_names = (
        "debug_displayed_effective_cost_min_confidence",
        "debug_preview_delta_min_confidence",
        "debug_resource_absolute_min_confidence",
    )
    explicit_transaction_thresholds = all(
        name in raw and raw[name] is not None for name in transaction_threshold_names
    )
    mode = params.get("live_decision_mode")
    if type(mode) is not str or mode not in {
        "observe",
        "selection_observe",
        "selection_reset",
        "status_observe", "status_reset", "item_status_observe", "item_status_full", "drink_observe", "drink_slots_observe", "drink_reset", "drink_slot_occupancy", "drink_inventory_observe", "turn_information_observe", "inventory_observe", "inventory_items_observe", "inventory_customization_observe", "inventory_card_state_observe", "opening_state_observe", "opening_shadow_transaction", "opening_execute",
        "shadow_transaction",
        "debug_execute",
    }:
        raise ValueError("unsupported_live_decision_mode")
    hif_profile = params.get('hif_policy_profile', 'joint-search')
    if hif_profile not in ('joint-search', 'hybrid-ppo27502-adaptive-v1',
                           'hybrid-ppo27502-visible-d-trial-v1', 'hif-mcts-v22-trial',
                           *FORMAL_PROFILES):
        raise ValueError('unsupported_hif_policy_profile')
    if hif_profile != 'joint-search' and mode not in ('opening_shadow_transaction', 'opening_execute'):
        raise ValueError('HIF candidate requires native complete-state mode')
    provider = params.get("provider", "weighted-policy")
    if type(provider) is not str or provider != "weighted-policy":
        raise ValueError("unsupported_live_decision_provider")
    preset = params.get("preset", "conservative-basic-v1")
    if type(preset) is not str or not preset:
        raise ValueError("invalid_live_decision_preset")

    params["debug_identity_threshold"] = _strict_finite_number(
        params,
        "debug_identity_threshold",
        default=0.70,
        lower=0.0,
        upper=1.0,
    )
    params["debug_identity_min_margin"] = _strict_finite_number(
        params,
        "debug_identity_min_margin",
        default=0.0,
        lower=0.0,
        upper=1.0,
    )
    params["provider_timeout_ms"] = _strict_finite_number(
        params,
        "provider_timeout_ms",
        default=None,
        lower=0.001,
        upper=300_000.0 if hif_profile in FORMAL_PROFILES else 60_000.0,
    )
    params["max_snapshot_age_ms"] = _strict_finite_number(
        params,
        "max_snapshot_age_ms",
        default=None,
        lower=0.001,
        upper=1_000.0 if hif_profile in FORMAL_PROFILES else 60_000.0,
    )
    params["minimum_identity_confidence"] = _strict_finite_number(
        params,
        "minimum_identity_confidence",
        default=None,
        lower=0.0,
        upper=1.0,
    )
    for threshold_name in transaction_threshold_names:
        params[threshold_name] = _strict_finite_number(
            params,
            threshold_name,
            default=None,
            lower=0.01,
            upper=1.0,
        )

    allowed_unknown = params.get("allowed_unknown_fields", ())
    if not isinstance(allowed_unknown, (list, tuple)):
        raise ValueError("allowed_unknown_fields must be an array")
    if any(type(value) is not str for value in allowed_unknown):
        raise ValueError("allowed_unknown_fields must contain strings")
    if len(set(allowed_unknown)) != len(allowed_unknown):
        raise ValueError("allowed_unknown_fields must not contain duplicates")
    if not set(allowed_unknown).issubset(LIVE_SOFT_UNKNOWN_FIELDS):
        raise ValueError("allowed_unknown_fields contains a hard safety field")
    params["allowed_unknown_fields"] = tuple(allowed_unknown)

    expected_plan = params.get("expected_plan")
    if expected_plan is not None and (
        type(expected_plan) is not str
        or expected_plan not in {"sense", "logic", "anomaly"}
    ):
        raise ValueError("invalid_expected_plan")
    if mode in {"selection_observe", "selection_reset", "status_observe", "status_reset", "item_status_observe", "item_status_full", "drink_observe", "drink_slots_observe", "drink_reset", "drink_slot_occupancy", "drink_inventory_observe", "turn_information_observe", "inventory_observe", "inventory_items_observe", "inventory_customization_observe", "inventory_card_state_observe", "opening_state_observe", "opening_shadow_transaction", "opening_execute", "shadow_transaction", "debug_execute"}:
        required_thresholds = (
            "provider_timeout_ms",
            "max_snapshot_age_ms",
            "minimum_identity_confidence",
        )
        if any(params[name] is None for name in required_thresholds):
            raise ValueError("input-capable mode requires approved runtime thresholds")
        if not explicit_identity_thresholds:
            raise ValueError("input-capable mode requires approved identity thresholds")
        if expected_plan is None:
            raise ValueError("input-capable mode requires an expected plan")
    if (
        mode in {"shadow_transaction", "debug_execute", "opening_shadow_transaction", "opening_execute"}
        and not explicit_transaction_thresholds
    ):
        raise ValueError("transaction mode requires approved resource thresholds")
    expected_initial_hand_count = params.get("debug_expected_initial_hand_count")
    if mode in {"shadow_transaction", "debug_execute", "opening_shadow_transaction", "opening_execute"}:
        if (
            "debug_expected_initial_hand_count" not in raw
            or type(expected_initial_hand_count) is not int
            or expected_initial_hand_count != 3
        ):
            raise ValueError(
                "transaction mode requires debug_expected_initial_hand_count=3"
            )
    elif expected_initial_hand_count is not None:
        raise ValueError(
            "debug_expected_initial_hand_count is unavailable outside transaction mode"
        )
    params["debug_expected_initial_hand_count"] = expected_initial_hand_count
    require_foreground_window = params.get("debug_require_foreground_window")
    if mode in {"debug_execute", "opening_execute"}:
        if (
            "debug_require_foreground_window" not in raw
            or type(require_foreground_window) is not bool
        ):
            raise ValueError(
                "debug_execute requires explicit debug_require_foreground_window boolean"
            )
    elif "debug_require_foreground_window" in raw:
        raise ValueError(
            "debug_require_foreground_window is unavailable outside debug_execute"
        )
    return params


def _is_process_elevated() -> bool:
    """Return true only when the Python Agent process itself is elevated."""

    if os.name != "nt":
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        return False


def _validate_live_input_clock_window(
    input_deadline: float | None,
    authorization_floor: float | None,
) -> tuple[bool, float | None]:
    """Advance a monotonic input-authorization floor or fail closed."""

    if (
        type(input_deadline) not in {int, float}
        or type(authorization_floor) not in {int, float}
        or not math.isfinite(float(input_deadline))
        or not math.isfinite(float(authorization_floor))
        or float(authorization_floor) >= float(input_deadline)
    ):
        return True, None
    try:
        observed = time.perf_counter()
    except Exception:
        return True, None
    if type(observed) not in {int, float} or not math.isfinite(float(observed)):
        return True, None
    observed_value = float(observed)
    if not (
        float(authorization_floor) < observed_value < float(input_deadline)
    ):
        return True, None
    return False, observed_value


def _input_deadline_expired(
    input_deadline: float | None,
    authorization_floor: float | None = None,
) -> bool:
    # Preserve the read-only legacy helper's explicit "no deadline" mode.
    # All input-authorizing transaction paths call the strict validator with
    # a required refresh/preflight floor instead of using this exemption.
    if input_deadline is None and authorization_floor is None:
        return False
    expired, _observed = _validate_live_input_clock_window(
        input_deadline,
        authorization_floor,
    )
    return expired


class _Win32Rect(ctypes.Structure):
    _fields_ = (
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    )


def _strict_positive_decimal_environment(name: str, *, upper: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise ValueError(f"{name}_missing")
    if re.fullmatch(r"[1-9][0-9]*", raw) is None:
        raise ValueError(f"{name}_invalid")
    value = int(raw)
    if value > upper:
        raise ValueError(f"{name}_invalid")
    return value


def _win32_user32() -> Any:
    if os.name != "nt":
        raise OSError("foreground_window_gate_is_windows_only")
    user32 = ctypes.windll.user32
    hwnd = ctypes.c_void_p
    bool_result = ctypes.c_int
    user32.IsWindow.argtypes = [hwnd]
    user32.IsWindow.restype = bool_result
    user32.IsWindowVisible.argtypes = [hwnd]
    user32.IsWindowVisible.restype = bool_result
    user32.IsIconic.argtypes = [hwnd]
    user32.IsIconic.restype = bool_result
    user32.GetWindowThreadProcessId.argtypes = [
        hwnd,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    user32.GetWindowThreadProcessId.restype = ctypes.c_uint32
    user32.GetClassNameW.argtypes = [
        hwnd,
        ctypes.POINTER(ctypes.c_wchar),
        ctypes.c_int,
    ]
    user32.GetClassNameW.restype = ctypes.c_int
    user32.GetWindowTextLengthW.argtypes = [hwnd]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [
        hwnd,
        ctypes.POINTER(ctypes.c_wchar),
        ctypes.c_int,
    ]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetClientRect.argtypes = [hwnd, ctypes.POINTER(_Win32Rect)]
    user32.GetClientRect.restype = bool_result
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = hwnd
    user32.SetForegroundWindow.argtypes = [hwnd]
    user32.SetForegroundWindow.restype = bool_result
    return user32


def _inspect_win32_window(user32: Any, hwnd: int) -> dict[str, Any]:
    facts = {
        "hwnd": hwnd,
        "pid": None,
        "window_class": None,
        "window_title": None,
        "client_size": None,
        "is_window": False,
        "is_visible": False,
        "is_iconic": None,
    }
    if hwnd <= 0 or not bool(user32.IsWindow(hwnd)):
        return facts
    facts["is_window"] = True
    facts["is_visible"] = bool(user32.IsWindowVisible(hwnd))
    facts["is_iconic"] = bool(user32.IsIconic(hwnd))
    process_id = ctypes.c_uint32()
    if int(user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))) != 0:
        facts["pid"] = int(process_id.value)
    class_name = ctypes.create_unicode_buffer(256)
    if int(user32.GetClassNameW(hwnd, class_name, len(class_name))) > 0:
        facts["window_class"] = class_name.value
    title_length = int(user32.GetWindowTextLengthW(hwnd))
    if title_length >= 0:
        title = ctypes.create_unicode_buffer(title_length + 1)
        if title_length == 0 or int(
            user32.GetWindowTextW(hwnd, title, len(title))
        ) > 0:
            facts["window_title"] = title.value
    rect = _Win32Rect()
    if bool(user32.GetClientRect(hwnd, ctypes.byref(rect))):
        facts["client_size"] = (
            int(rect.right - rect.left),
            int(rect.bottom - rect.top),
        )
    return facts


def _foreground_gate_event(
    *,
    phase: str = "foreground_gate",
    preparation_stage: str | None = None,
    activation_allowed: bool = True,
) -> dict[str, Any]:
    event = {
        "phase": phase,
        "click_dispatched": False,
        "foreground_gate_required": True,
        "activation_allowed": activation_allowed,
        "activation_attempted": False,
        "activation_accepted": False,
        "activation_reason": None,
        "activation_api_returned": None,
        "expected_hwnd": None,
        "expected_pid": None,
        "expected_window_class": DEBUG_TARGET_WINDOW_CLASS,
        "expected_window_title": DEBUG_TARGET_WINDOW_TITLE,
        "expected_client_size": [720, 1280],
        "actual_hwnd": None,
        "actual_pid": None,
        "actual_window_class": None,
        "actual_window_title": None,
        "actual_client_size": None,
        "actual_is_visible": None,
        "actual_is_iconic": None,
        "validated_target_hwnd": None,
        "validated_target_pid": None,
        "validated_target_window_class": None,
        "validated_target_window_title": None,
        "validated_target_client_size": None,
        "validated_target_is_visible": None,
        "validated_target_is_iconic": None,
        "window_focus_may_have_changed": False,
        "input_may_have_been_dispatched": False,
        "accepted": False,
    }
    if preparation_stage is not None:
        event["preparation_stage"] = preparation_stage
    return event


def _debug_foreground_window_gate(
    input_deadline: float | None,
    *,
    authorization_floor: float | None = None,
    allow_activation: bool = True,
    phase: str = "foreground_gate",
    preparation_stage: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Validate and, if needed, activate the runner-sealed DMM target."""

    event = _foreground_gate_event(
        phase=phase,
        preparation_stage=preparation_stage,
        activation_allowed=allow_activation,
    )
    try:
        expected_hwnd = _strict_positive_decimal_environment(
            DEBUG_TARGET_HWND_ENV,
            upper=(1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1,
        )
        expected_pid = _strict_positive_decimal_environment(
            DEBUG_TARGET_PID_ENV,
            upper=(1 << 32) - 1,
        )
    except ValueError as error:
        event["activation_reason"] = str(error)
        return False, event
    event.update({"expected_hwnd": expected_hwnd, "expected_pid": expected_pid})

    try:
        user32 = _win32_user32()
        target = _inspect_win32_window(user32, expected_hwnd)
    except (
        AttributeError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
        ctypes.ArgumentError,
    ) as error:
        event.update(
            {
                "activation_reason": "foreground_window_api_unavailable",
                "activation_error_type": type(error).__name__,
            }
        )
        return False, event
    event.update(
        {
            "actual_hwnd": target["hwnd"],
            "actual_pid": target["pid"],
            "actual_window_class": target["window_class"],
            "actual_window_title": target["window_title"],
            "actual_client_size": target["client_size"],
            "actual_is_visible": target["is_visible"],
            "actual_is_iconic": target["is_iconic"],
            "validated_target_hwnd": target["hwnd"],
            "validated_target_pid": target["pid"],
            "validated_target_window_class": target["window_class"],
            "validated_target_window_title": target["window_title"],
            "validated_target_client_size": target["client_size"],
            "validated_target_is_visible": target["is_visible"],
            "validated_target_is_iconic": target["is_iconic"],
        }
    )
    target_rejections = (
        (not target["is_window"], "debug_target_hwnd_is_not_window"),
        (target["pid"] != expected_pid, "debug_target_pid_mismatch"),
        (
            target["window_class"] != DEBUG_TARGET_WINDOW_CLASS,
            "debug_target_window_class_mismatch",
        ),
        (
            target["window_title"] != DEBUG_TARGET_WINDOW_TITLE,
            "debug_target_window_title_mismatch",
        ),
        (target["client_size"] != (720, 1280), "debug_target_client_size_mismatch"),
        (target["is_visible"] is not True, "debug_target_window_not_visible"),
        (target["is_iconic"] is not False, "debug_target_window_is_iconic"),
    )
    for rejected, reason in target_rejections:
        if rejected:
            event["activation_reason"] = reason
            return False, event
    if input_deadline is not None or authorization_floor is not None:
        deadline_expired, observed_floor = _validate_live_input_clock_window(
            input_deadline,
            authorization_floor,
        )
        if deadline_expired:
            event["activation_reason"] = (
                "snapshot_expired_before_foreground_activation"
            )
            return False, event
        authorization_floor = observed_floor
        event["authorization_floor"] = authorization_floor

    try:
        foreground_before = int(user32.GetForegroundWindow() or 0)
        if foreground_before != expected_hwnd and allow_activation:
            event["activation_attempted"] = True
            event["window_focus_may_have_changed"] = True
            event["activation_api_returned"] = bool(
                user32.SetForegroundWindow(expected_hwnd)
            )
        foreground_after = int(user32.GetForegroundWindow() or 0)
        actual = _inspect_win32_window(user32, foreground_after)
    except (
        AttributeError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
        ctypes.ArgumentError,
    ) as error:
        event.update(
            {
                "activation_reason": "foreground_window_api_failed",
                "activation_error_type": type(error).__name__,
            }
        )
        return False, event
    event.update(
        {
            "actual_hwnd": actual["hwnd"],
            "actual_pid": actual["pid"],
            "actual_window_class": actual["window_class"],
            "actual_window_title": actual["window_title"],
            "actual_client_size": actual["client_size"],
            "actual_is_visible": actual["is_visible"],
            "actual_is_iconic": actual["is_iconic"],
        }
    )
    if input_deadline is not None or authorization_floor is not None:
        deadline_expired, observed_floor = _validate_live_input_clock_window(
            input_deadline,
            authorization_floor,
        )
        if deadline_expired:
            event["activation_reason"] = (
                "snapshot_expired_after_foreground_activation"
            )
            return False, event
        authorization_floor = observed_floor
        event["authorization_floor"] = authorization_floor
    if event["activation_attempted"] and event["activation_api_returned"] is not True:
        event["activation_reason"] = "set_foreground_window_failed"
        return False, event
    if foreground_after != expected_hwnd:
        event["activation_reason"] = (
            "foreground_window_mismatch_after_activation"
            if allow_activation
            else "foreground_window_mismatch_activation_forbidden"
        )
        return False, event
    if (
        actual["pid"] != expected_pid
        or actual["window_class"] != DEBUG_TARGET_WINDOW_CLASS
        or actual["window_title"] != DEBUG_TARGET_WINDOW_TITLE
        or actual["client_size"] != (720, 1280)
        or actual["is_visible"] is not True
        or actual["is_iconic"] is not False
    ):
        event["activation_reason"] = "foreground_window_identity_changed"
        return False, event
    event.update(
        {
            "activation_accepted": True,
            "activation_reason": (
                "foreground_activation_verified"
                if event["activation_attempted"]
                else "target_already_foreground"
            ),
            "accepted": True,
        }
    )
    return True, event


def _debug_foreground_window_preparation() -> tuple[bool, list[dict[str, Any]]]:
    """Activate once, then keep the sealed target stable for 500 ms."""

    events: list[dict[str, Any]] = []
    accepted, initial = _debug_foreground_window_gate(
        None,
        allow_activation=True,
        phase="foreground_preparation",
        preparation_stage="initial",
    )
    initial["settle_target_ms"] = 0
    initial["click_dispatched"] = False
    initial["input_may_have_been_dispatched"] = False
    events.append(initial)
    pending_activation = (
        not accepted
        and initial.get("activation_attempted") is True
        and initial.get("activation_api_returned") is True
        and initial.get("actual_hwnd") == 0
        and initial.get("activation_reason")
        == "foreground_window_mismatch_after_activation"
    )
    if not accepted and not pending_activation:
        return False, events
    stable_identity_fields = (
        "expected_hwnd",
        "expected_pid",
        "validated_target_hwnd",
        "validated_target_pid",
        "validated_target_window_class",
        "validated_target_window_title",
        "validated_target_client_size",
        "validated_target_is_visible",
        "validated_target_is_iconic",
        "actual_hwnd",
        "actual_pid",
        "actual_window_class",
        "actual_window_title",
        "actual_client_size",
        "actual_is_visible",
        "actual_is_iconic",
    )
    stable_identity = {
        field: initial.get(field) for field in stable_identity_fields
    }
    stages = (("midpoint", 250), ("settled", 500))
    if pending_activation:
        # Windows may briefly report no foreground HWND after a successful
        # activation. Recheck once without activating again, against the
        # already verified target; only then begin the full 500 ms stability
        # interval. Keep the failed immediate observation unchanged.
        for field in stable_identity_fields:
            if field.startswith("actual_"):
                stable_identity[field] = initial.get(
                    field.replace("actual_", "validated_target_", 1)
                )
        stages = (("activation_settle", 250), ("midpoint", 500), ("settled", 750))

    for stage, settle_target_ms in stages:
        time.sleep(DEBUG_FOREGROUND_PREPARATION_STEP_SECONDS)
        accepted, verification = _debug_foreground_window_gate(
            None,
            allow_activation=False,
            phase="foreground_preparation",
            preparation_stage=stage,
        )
        verification["settle_target_ms"] = settle_target_ms
        verification["click_dispatched"] = False
        verification["input_may_have_been_dispatched"] = False
        if verification.get("activation_attempted") is True:
            accepted = False
            verification.update(
                {
                    "accepted": False,
                    "activation_accepted": False,
                    "activation_reason": (
                        "foreground_reverification_attempted_activation"
                    ),
                }
            )
        elif verification.get("activation_allowed") is not False:
            accepted = False
            verification.update(
                {
                    "accepted": False,
                    "activation_accepted": False,
                    "activation_reason": (
                        "foreground_reverification_activation_not_forbidden"
                    ),
                }
            )
        elif accepted and any(
            verification.get(field) != expected
            for field, expected in stable_identity.items()
        ):
            accepted = False
            verification.update(
                {
                    "accepted": False,
                    "activation_accepted": False,
                    "activation_reason": (
                        "foreground_preparation_target_identity_changed"
                    ),
                }
            )
        events.append(verification)
        if not accepted:
            return False, events
    return True, events


def _adapt_hif_capture_context(context):
    """Bind the observed client, then share one normalized HIF coordinate space."""
    from live_decision.hif_capture_adapter import adapt_hif_context

    size = getattr(context, 'gkh_hif_client_size', None)
    if size is None:
        info = context.tasker.controller.info
        if (not isinstance(info, dict) or info.get('type') != 'win32'
                or type(info.get('hwnd')) is not int):
            raise ValueError('hif_capture_controller_unresolved')
        facts = _inspect_win32_window(_win32_user32(), info['hwnd'])
        if (not facts['is_window'] or facts['window_title'] != 'gakumas'
                or facts['window_class'] != 'UnityWndClass'):
            raise ValueError('hif_capture_window_unresolved')
        size = facts['client_size']
        context.gkh_hif_client_size = size
    if calibrated_live_client_size(size) is None:
        raise ValueError('hif_capture_client_uncalibrated')
    return adapt_hif_context(context)


def _live_field_roi(context, name: str, roi: list[int], image=None) -> list[int]:
    """Use the same scene geometry in direct and batched field recognition."""
    if isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext):
        if name == "turns_remaining":
            # Keep the centre numeral, excluding HIF's coloured turn ring.
            return [40, 68, 43, 42]
        if name == "score_multiplier":
            # HIF's coloured turn ring intrudes into the old NIA crop and
            # can become an extra leading digit. Keep the percentage only.
            return [110, 82, 115, 37]
        if name == "stamina":
            # Share the complete numeric crop validated by HIF recovery;
            # stage lights outside it can be mistaken for punctuation.
            return [573, 212, 49, 32]
        if name == "shield":
            if (tuple(getattr(context, 'gkh_hif_client_size', ())) in
                    {(540, 960), (541, 961), (541, 962), (542, 962)}
                    and tuple(getattr(image, 'shape', ())) == (1280, 720, 3)):
                # Some resampled HUDs place the badge left of the old crop.
                # Expand from its observed cyan face; never remove any pixel
                # of the original full ROI or derive a digit from its colour.
                import cv2
                hsv = cv2.cvtColor(image[175:209, 640:715], cv2.COLOR_BGR2HSV)
                cyan = cv2.inRange(hsv, (80, 80, 150), (100, 255, 255))
                _, _, parts, _ = cv2.connectedComponentsWithStats(cyan)
                faces = [row for row in parts[1:] if 648 <= 640+row[0] < 660
                    and 1 <= row[1] <= 5 and 25 <= row[2] <= 60
                    and 20 <= row[3] <= 28 and 685 <= 640+row[0]+row[2] <= 715
                    and row[4] >= 100]
                if len(faces) == 1:
                    left = 640+int(faces[0][0])
                    return [left, 175, 715-left, 34]
            if calibrated_live_client_size(getattr(context, "gkh_hif_client_size", ())) == (480, 853):
                # At the native 480px layout the badge sits farther right.
                # Preserve the whole digit after official long-side scaling.
                return [660, 177, 55, 31]
            # The old NIA crop ends at y=200, inside HIF's digit. Exclude
            # the stamina bar and outer badge edge, preserving its face.
            return [660, 175, 55, 34]
    return list(roi)


def _ocr_field(
    context: Context,
    image: Any,
    name: str,
    roi: list[int],
    expected: str,
) -> dict[str, Any]:
    node_name = f"Task200Visual{name}"
    hif_run = getattr(context, "gkh_hif_live_context", None)
    if name == "stamina" and isinstance(hif_run, HifLiveRunContext):
        # HIF's heart overlaps the old crop and was read as an extra digit
        # (242 instead of 42). This numeric-only area is calibrated for the
        # current one/two-digit HIF layout, not for three-digit stamina.
        if hif_run.max_stamina >= 100:
            return {"accepted": False, "value": None, "confidence": 0.0,
                    "reason": "hif_three_digit_stamina_uncalibrated"}
        expected = r"^\d{1,2}$"
    roi = _live_field_roi(context, name, roi, image)
    # The accepted HIF rank layout supplies this complete single score area.
    # 1250 text detection kept only tiny internal glyphs (G/0); recognize the
    # calibrated whole value directly, as for the other fixed HUD numerals.
    whole_hif_score = (name == 'score' and isinstance(hif_run, HifLiveRunContext)
        and (roi == [365, 120, 135, 50]
             and calibrated_live_client_size(getattr(context, 'gkh_hif_client_size', ())) == (720, 1280)
             # 1906's fourth column exposes the same complete numeric area;
             # detection dropped its last digit while assigning high confidence.
             or roi in ([434, 120, 135, 50],[294, 120, 126, 50])
             and tuple(getattr(context, 'gkh_hif_client_size', ())) in
                 {(540, 960), (541, 961), (541, 962), (542, 962)}
             and tuple(getattr(image, 'shape', ())) == (1280, 720, 3)))
    resampled_shield = (name == 'shield' and isinstance(hif_run, HifLiveRunContext)
        and (calibrated_live_client_size(getattr(context, 'gkh_hif_client_size', ())) == (480, 853)
             or tuple(getattr(context, 'gkh_hif_client_size', ())) in
                 {(540, 960), (541, 961), (541, 962), (542, 962)})
        and tuple(getattr(image, 'shape', ())) == (1280, 720, 3)
        and (roi[1], roi[3]) in {(175, 34), (177, 31)} and 648 <= roi[0] <= 660
        and roi[0] + roi[2] == 715)
    stamina_contrast_detection = (name == 'stamina' and isinstance(hif_run, HifLiveRunContext)
        and calibrated_live_client_size(getattr(context, 'gkh_hif_client_size', ())) == (480, 853)
        and tuple(getattr(image, 'shape', ())) == (1280, 720, 3)
        and roi == [573, 212, 49, 32])
    shield_contrast_detection = False
    if resampled_shield:
        x,y,width,height = roi
        luminance = image[y:y+height,x:x+width].min(axis=2)
        # Choose the same complete-region detector before reading its text.
        # The normal-position badge has the same coloured-background failure
        # as a shifted badge. Flat/empty regions retain their existing path.
        shield_contrast_detection = (roi[0] < 660 or
            int(luminance.max())-int(luminance.min()) >= 128
            and int((luminance <= 100).sum()) >= 6
            and int((luminance >= 230).sum()) >= 6)
    recognition_image = image
    if shield_contrast_detection or stamina_contrast_detection:
        # Stage pixels at the edge of a resampled numeric region can become
        # punctuation in whole-crop recognition (Full155 stamina: "37.").
        # Reuse complete-region colour suppression and text detection;
        # neither expected values nor text rewriting enter this read.
        recognition_image = image.copy()
        x, y, width, height = roi
        recognition_image[y:y+height, x:x+width] = image[y:y+height, x:x+width].min(axis=2)[:, :, None]
    detail = context.run_recognition(
        node_name,
        recognition_image,
        pipeline_override={
            node_name: {
                "recognition": "OCR",
                "roi": roi,
                "expected": expected,
                **({"only_rec": True} if whole_hif_score or (not stamina_contrast_detection and
                    name in {"turns_remaining", "score_multiplier", "stamina"}
                    and isinstance(hif_run, HifLiveRunContext)) else {}),
            }
        },
    )
    results = (
        list(getattr(detail, "filtered_results", ()) or ())
        if detail and detail.hit
        else []
    )
    observed = _parse_live_field_ocr_results(results, name=name, expected=expected)
    raw = list(getattr(detail, 'all_results', ()) or ()) if detail else []
    split_contrast_shield = False
    weak_compact_shield_digit = False
    if shield_contrast_detection or stamina_contrast_detection:
        if (len(raw) != 1 or len(results) != 1 or not _valid_ocr_result(raw[0])
                or not (roi[0] <= raw[0].box[0] < raw[0].box[0]+raw[0].box[2] <= roi[0]+roi[2]
                    and roi[1] <= raw[0].box[1] < raw[0].box[1]+raw[0].box[3] <= roi[1]+roi[3])):
            observed.update(accepted=False, value=None, confidence=0.,
                reason=('hif_stamina_detection_unresolved' if stamina_contrast_detection
                    else 'hif_shifted_shield_detection_unresolved'))
        observed = dict(observed, acquisition_mode=('hif_resampled_stamina_contrast_detection'
            if stamina_contrast_detection else 'hif_shifted_shield_contrast_detection'
                if roi[0] < 660 else 'hif_resampled_shield_contrast_detection'),
            roi=roi, contrast_basis='minimum_channel',
            raw_results=[dict(text=getattr(row, 'text', None), confidence=getattr(row, 'score', None),
                box=list(getattr(row, 'box', ()) or ())) for row in raw])
        # The calibrated detector can split a single badge into glyphs too.
        # Reuse the existing independent whole-ROI read; never join the text
        # or use the other frames' value to resolve this frame.
        split_contrast_shield = shield_contrast_detection and _hif_shield_text_fragments(raw, roi)
        # 0026: a single detected digit can be weak over this moving cyan bar.
        # Use the existing whole-badge reader on its calibrated complete ROI.
        weak_compact_shield_digit = (shield_contrast_detection
            and tuple(getattr(context, 'gkh_hif_client_size', ())) == (480, 853)
            and roi == [660, 177, 55, 31] and observed.get('accepted') is True
            and 0 <= observed['confidence'] < .95 and re.fullmatch(r'[0-9]', observed['text']) is not None)
        if not (split_contrast_shield or weak_compact_shield_digit):
            return observed
    if (name == 'shield' and isinstance(hif_run, HifLiveRunContext)
            and tuple(getattr(context, 'gkh_hif_client_size', ())) in
                {(540, 960), (541, 961), (541, 962), (542, 962)}
            and observed.get('reason') == 'live_field_ocr_ambiguous'):
        badge = _hif_shield_numeric_badge_roi(image, roi, allow_filled_bar=True)
        retained = _hif_shield_bar_owned_detection(image, roi, raw, badge)
        if retained is not None:
            # A detector rectangle can include the bar and the badge's outer
            # cyan edge. Only independently proven non-ink pixels may leave
            # the competition; the complete badge still needs its own read.
            badge_name = 'Task200VisualshieldBarOwnedBadge'
            hit = context.run_recognition(badge_name, image, pipeline_override={badge_name: {
                'recognition': 'OCR', 'roi': badge['roi'], 'expected': expected, 'only_rec': True,
            }})
            badge_all = list(getattr(hit, 'all_results', ()) or ()) if hit else []
            badge_rows = list(getattr(hit, 'filtered_results', ()) or ()) if hit and hit.hit else []
            numeric = _parse_live_field_ocr_results(badge_rows, name=name, expected=expected)
            original = dict(observed, raw_results=[dict(text=row.text,
                confidence=row.score, box=list(row.box)) for row in raw])
            kept = _parse_live_field_ocr_results([retained], name=name, expected=expected)
            if (len(badge_all) == len(badge_rows) == 1 and numeric.get('accepted') is True
                    and .95 <= numeric.get('confidence', 0) <= 1
                    and tuple(badge_rows[0].box) == tuple(badge['roi'])
                    and kept.get('accepted') is True and kept.get('confidence', 0) >= .95
                    and numeric.get('value') == kept.get('value')):
                return dict(numeric, acquisition_mode='hif_shield_bar_owned_badge_reread',
                    roi=badge['roi'], badge_boundary=badge, original_observation=original)
            return dict(original, badge_observation=numeric, badge_boundary=badge,
                reason='hif_shield_bar_owned_badge_unresolved')
    split_shield = split_contrast_shield or (name == 'shield' and isinstance(hif_run, HifLiveRunContext)
        and calibrated_live_client_size(getattr(context, 'gkh_hif_client_size', ())) == (720, 1280)
        and _hif_shield_text_fragments(raw, roi))
    weak_single_shield = (name == 'shield' and isinstance(hif_run, HifLiveRunContext)
        and calibrated_live_client_size(getattr(context, 'gkh_hif_client_size', ())) in
            {(720, 1280), (480, 853)}
        and len(raw) == len(results) == 1 and observed.get('accepted') is True
        and _valid_ocr_result(raw[0]) and type(getattr(raw[0], 'text', None)) is str
        and re.fullmatch(expected, raw[0].text) is not None
        and 0 <= observed.get('confidence', 0) < .95
        and roi[0] <= raw[0].box[0] < raw[0].box[0]+raw[0].box[2] <= roi[0]+roi[2]
        and roi[1] <= raw[0].box[1] < raw[0].box[1]+raw[0].box[3] <= roi[1]+roi[3])
    missing_single_shield = (name == 'shield' and isinstance(hif_run, HifLiveRunContext)
        and calibrated_live_client_size(getattr(context, 'gkh_hif_client_size', ())) == (720, 1280)
        and len(raw) == 1 and not results and observed.get('reason') == 'live_field_ocr_missing'
        and _valid_ocr_result(raw[0]) and type(getattr(raw[0], 'text', None)) is str
        and re.fullmatch(expected, raw[0].text) is None
        and roi[0] <= raw[0].box[0] < raw[0].box[0]+raw[0].box[2] <= roi[0]+roi[2]
        and roi[1] <= raw[0].box[1] < raw[0].box[1]+raw[0].box[3] <= roi[1]+roi[3])
    if (name == 'shield' and isinstance(hif_run, HifLiveRunContext)
            and (split_shield or weak_single_shield or missing_single_shield or (
                calibrated_live_client_size(getattr(context, 'gkh_hif_client_size', ())) == (480, 853)
                and observed.get('reason') == 'live_field_ocr_missing'))):
        # Native 1023: text detection clipped the badge's 9 into b / Q.
        # Native 1202/1209: detection splits one numeric badge into glyphs;
        # regex filtering may hide a misread letter and accept only a prefix.
        # 1250: a single weak detector crop also trims the badge's right/bottom
        # pixel. The same complete ROI must independently agree on its value.
        # 1924: the calibrated 480px viewport has the same weak single-crop
        # failure. Its normalized badge uses the same complete ROI and gate.
        # Read this complete single-value ROI once, without concatenation or
        # predicted values. Competing complete readings cannot enter here.
        original = dict(observed, raw_results=[dict(text=getattr(row, 'text', None),
            confidence=getattr(row, 'score', None), box=list(getattr(row, 'box', ()) or ())) for row in raw])
        reread_name = 'Task200VisualshieldCompleteRoi'
        reread = context.run_recognition(reread_name,
            recognition_image if split_contrast_shield else image, pipeline_override={reread_name: {
            'recognition':'OCR', 'roi':roi, 'expected':expected, 'only_rec':True,
        }})
        all_rows = list(getattr(reread, 'all_results', ()) or ()) if reread else []
        filtered = list(getattr(reread, 'filtered_results', ()) or ()) if reread and reread.hit else []
        complete = _parse_live_field_ocr_results(filtered, name=name, expected=expected)
        if (len(all_rows) == 1 and len(filtered) <= 1
                and (complete.get('accepted') is not True or complete.get('confidence', 0) < .95)
                and _valid_ocr_result(all_rows[0]) and tuple(all_rows[0].box) == tuple(roi)
                and calibrated_live_client_size(getattr(context, 'gkh_hif_client_size', ())) == (720, 1280)):
            resampled = tuple(getattr(context, 'gkh_hif_client_size', ())) in {
                (540, 960), (541, 961), (541, 962), (542, 962)}
            badge = _hif_shield_numeric_badge_roi(
                image, roi, **({'allow_filled_bar': True} if resampled else {}))
            if badge is not None:
                # 1267's cap is independently visible outside the numeric
                # badge. Read the entire badge; never remove D/0 from OCR text.
                badge_name = 'Task200VisualshieldNumericBadge'
                badge_image = image
                if resampled:
                    import cv2
                    badge_image = image.copy()
                    x, y, width, height = roi
                    badge_image[y:y+height, x:x+width] = cv2.cvtColor(
                        image[y:y+height, x:x+width].min(axis=2), cv2.COLOR_GRAY2BGR)
                hit = context.run_recognition(badge_name, badge_image, pipeline_override={badge_name: {
                    'recognition': 'OCR', 'roi': badge['roi'], 'expected': expected, 'only_rec': True,
                }})
                badge_all = list(getattr(hit, 'all_results', ()) or ()) if hit else []
                badge_rows = list(getattr(hit, 'filtered_results', ()) or ()) if hit and hit.hit else []
                numeric = _parse_live_field_ocr_results(badge_rows, name=name, expected=expected)
                if (resampled and numeric.get('accepted') is True
                        and any(previous.get('accepted') is True
                                and numeric.get('value') != previous.get('value')
                                for previous in (original, complete))):
                    return dict(original, accepted=False, value=None, confidence=0.,
                        reason='hif_shield_numeric_badge_conflict', badge_observation=numeric,
                        complete_roi_reread=complete, badge_boundary=badge)
                if (len(badge_all) == len(badge_rows) == 1 and numeric.get('accepted') is True
                        and .95 <= numeric.get('confidence', 0) <= 1
                        and all(previous.get('accepted') is not True
                                or numeric.get('value') == previous.get('value')
                                for previous in (original, complete))):
                    return dict(numeric, acquisition_mode='hif_shield_numeric_badge_reread',
                        roi=badge['roi'], badge_boundary=badge, complete_roi_reread=complete,
                        original_observation=original,
                        **({'contrast_basis': 'minimum_channel'} if resampled else {}))
        if (len(all_rows) == len(filtered) == 1 and complete.get('accepted') is True
                and .95 <= complete.get('confidence', 0) <= 1
                and (not split_contrast_shield or tuple(all_rows[0].box) == tuple(roi))
                and (not weak_single_shield or complete.get('value') == original.get('value'))):
            return dict(complete, acquisition_mode='hif_shield_complete_roi_reread',
                        roi=roi, original_observation=original,
                        **({'contrast_basis': 'minimum_channel'} if split_contrast_shield else {}))
        complete_detection_needed = (
            missing_single_shield and raw[0].score < .95
            or weak_single_shield and tuple(raw[0].box) == tuple(roi)
            and complete.get('value') == original.get('value'))
        if (complete_detection_needed
                and len(all_rows) == len(filtered) == 1 and complete.get('accepted') is True
                and .85 <= complete.get('confidence', 0) < .95
                and tuple(all_rows[0].box) == tuple(roi)
                and tuple(getattr(context, 'gkh_hif_client_size', ())) in
                    {(540, 960), (541, 961), (541, 962), (542, 962)}
                and tuple(getattr(image, 'shape', ())) == (1280, 720, 3)):
            # The complete numeral is weak; the original detector found a
            # nonnumeric fragment or supplied the same whole-ROI read.
            # Reuse full-ROI colour suppression
            # with text detection, as for the shifted badge; all pixels stay.
            contrast_image = image.copy()
            x, y, width, height = roi
            contrast_image[y:y+height,x:x+width] = image[y:y+height,x:x+width].min(axis=2)[:, :, None]
            contrast_name = 'Task200VisualshieldCompleteContrastDetection'
            hit = context.run_recognition(contrast_name,contrast_image,pipeline_override={contrast_name:{
                'recognition':'OCR','roi':roi,'expected':expected}})
            detected_all = list(getattr(hit,'all_results',()) or ()) if hit else []
            detected_rows = list(getattr(hit,'filtered_results',()) or ()) if hit and hit.hit else []
            detected = _parse_live_field_ocr_results(detected_rows,name=name,expected=expected)
            if (len(detected_all) == len(detected_rows) == 1 and _valid_ocr_result(detected_all[0])
                    and detected.get('accepted') is True and detected.get('confidence',0) >= .95
                    and detected.get('value') == complete.get('value')
                    and x <= detected_all[0].box[0] < sum(detected_all[0].box[::2]) <= x+width
                    and y <= detected_all[0].box[1] < sum(detected_all[0].box[1::2]) <= y+height):
                return dict(detected,acquisition_mode='hif_shield_complete_contrast_detection',
                    roi=roi,contrast_basis='minimum_channel',complete_roi_reread=complete,
                    original_observation=original)
            return dict(original,complete_roi_reread=complete,
                complete_contrast_detection=dict(detected,raw_result_count=len(detected_all)),
                accepted=False,value=None,confidence=0.,reason='hif_shield_complete_detection_unresolved')
        contrast_diagnostic = None
        weak_rejected_complete = (
            len(all_rows) == 1 and not filtered
            and complete.get('accepted') is False
            and _valid_ocr_result(all_rows[0])
            and all_rows[0].score < .95
            and tuple(all_rows[0].box) == tuple(roi)
            and re.fullmatch(expected, all_rows[0].text) is None
        )
        complete_agrees_or_unreadable = (
            len(all_rows) == len(filtered) == 1
            and complete.get('accepted') is True
            and complete.get('value') == original.get('value')
        ) or weak_rejected_complete
        if (weak_single_shield and complete_agrees_or_unreadable
                and (weak_compact_shield_digit or tuple(getattr(context, 'gkh_hif_client_size', ())) in
                    {(540, 960), (541, 961), (541, 962), (542, 962)})
                and tuple(getattr(image, 'shape', ())) == (1280, 720, 3)):
            # The resampled cyan badge can obscure the white numeral. Keep
            # the complete ROI and suppress colour only, as the drink reader
            # already does; no stroke, cap, or disagreeing reading is removed.
            # A weak rejected whole-ROI glyph supplies no numeric value. Keep
            # it as evidence; the independent complete read must still agree.
            if weak_rejected_complete:
                complete = dict(complete, rejected_result=dict(text=all_rows[0].text,
                    confidence=all_rows[0].score, box=list(all_rows[0].box)))
            import cv2
            x, y, width, height = roi
            crop = cv2.cvtColor(image[y:y+height, x:x+width].min(axis=2), cv2.COLOR_GRAY2BGR)
            contrast_name = 'Task200VisualshieldCompleteContrast'
            hit = context.run_recognition(contrast_name, crop, pipeline_override={contrast_name: {
                'recognition': 'OCR', 'roi': [0, 0, width, height], 'expected': expected, 'only_rec': True,
            }})
            contrast_all = list(getattr(hit, 'all_results', ()) or ()) if hit else []
            contrast_rows = list(getattr(hit, 'filtered_results', ()) or ()) if hit and hit.hit else []
            contrast = _parse_live_field_ocr_results(contrast_rows, name=name, expected=expected)
            contrast_diagnostic = dict(contrast, raw_result_count=len(contrast_all))
            if (len(contrast_all) == len(contrast_rows) == 1 and contrast.get('accepted') is True
                    and tuple(contrast_rows[0].box) == (0, 0, width, height)
                    and .95 <= contrast.get('confidence', 0) <= 1
                    and contrast.get('value') == original.get('value')):
                return dict(contrast, box=list(roi), acquisition_mode='hif_shield_complete_contrast',
                    roi=roi, contrast_basis='minimum_channel', complete_roi_reread=complete,
                    original_observation=original)
        observed = dict(original, complete_roi_reread=complete)
        if contrast_diagnostic is not None:
            observed['complete_contrast_reread'] = contrast_diagnostic
        if split_shield or weak_single_shield:
            # A filtered prefix is not the badge's complete value, even if
            # it is independently numeric. Failed whole reads stay unknown.
            observed.update(accepted=False, value=None, confidence=0.0,
                reason='hif_shield_complete_roi_unresolved', original_observation=original)
    return observed


def _hif_shield_numeric_badge_roi(image, roi, *, allow_filled_bar=False):
    """Separate the cyan numeral badge from a full-height white bar cap.

    This only proves a crop. All digits still require independent OCR, and
    any additional white glyph left of the badge prevents trimming it.
    """
    import cv2
    import numpy as np

    if tuple(getattr(image, 'shape', ())) != (1280, 720, 3) or roi != [660, 175, 55, 34]:
        return None
    hsv = cv2.cvtColor(image[175:209, 660:715], cv2.COLOR_BGR2HSV)
    # The signed preview bubble starts below y=203. Its cyan edge can join
    # the badge in a later animation frame (1268); it is not part of this
    # boundary proof. OCR below still receives the full 34-pixel value ROI.
    cyan = cv2.inRange(hsv[:28], (80, 80, 150), (100, 255, 255))
    _, labels, components, _ = cv2.connectedComponentsWithStats(cyan)
    badges = [tuple(map(int, row)) for row in components[1:]
        if 0 < row[0] and 1 <= row[1] <= 5 and 12 <= row[2] <= 53
        and 18 <= row[3] <= 27 and 49 <= row[0]+row[2] <= 54 and row[4] >= 50]
    if len(badges) != 1:
        return None
    x, y, width, height, _ = badges[0]
    if max((cyan[row, x:x+width] > 0).sum() for row in range(y, y+min(5,height))) < .8*width:
        return None
    # The bar cap spans almost the complete height. A digit begins lower;
    # it may not be mistaken for this cap even if the cyan rim is damaged.
    white = cv2.inRange(hsv[:28, :x], (0, 0, 210), (180, 70, 255))
    _, _, parts, _ = cv2.connectedComponentsWithStats(white)
    # A highlight above the independently located badge is not a numeral.
    # Keep every component intersecting its vertical band, regardless of
    # area; even a tiny possible extra digit stroke must still reject.
    relevant = [row for row in parts[1:] if row[1] < y+height and row[1]+row[3] > y]
    if not relevant:
        # An unselected HUD has the same bar cap in cyan instead of the
        # selected-preview white highlight. It must be a separate, complete
        # component connected to the left bar edge, with no white numeral
        # stroke anywhere in the portion being excluded.
        caps = [tuple(map(int, row)) for row in components[1:]
            if row[0] == 0 and 0 < row[1] <= 3 and 24 <= row[3] <= 28
            and row[1]+row[3] == 28 and abs(row[0]+row[2]-x) <= 1]
        if len(caps) != 1:
            if not allow_filled_bar:
                return None
            # At a resampled, unselected HUD the coloured bar is thinner
            # than the old highlighted cap. Prove its full horizontal body
            # outside the numeric ROI, not a shortened digit-like fragment.
            wide_hsv = cv2.cvtColor(image[175:203, 600:715], cv2.COLOR_BGR2HSV)
            wide_cyan = cv2.inRange(wide_hsv, (80, 80, 150), (100, 255, 255))
            # The filled bar changes from green to cyan along its length.
            # Keep the original cyan mask throughout the numeric ROI.
            wide_cyan[:, :60] = cv2.inRange(wide_hsv[:, :60], (40, 80, 150), (100, 255, 255))
            _, wide_labels, wide_parts, _ = cv2.connectedComponentsWithStats(wide_cyan)
            candidates = []
            for label, part in enumerate(components[1:], 1):
                bx, by, bw, bh, area = map(int, part)
                if not (bx == 0 and 5 <= by <= 12 and 10 <= bh <= 20
                        and by+bh <= 26 and bw <= 14 and bw < x):
                    continue
                overlapping = set(map(int, wide_labels[:, 60:][labels == label])) - {0}
                if len(overlapping) != 1:
                    continue
                wide_label = overlapping.pop()
                wx, wy, ww, wh, _ = map(int, wide_parts[wide_label])
                if (wx == 0 and 60 < ww < 60+x and (wy, wh) == (by, bh)
                        and np.array_equal(wide_labels[:, 60:] == wide_label, labels == label)):
                    candidates.append((part, [600+wx, 175+wy, ww, wh]))
            if len(candidates) != 1:
                # 1358 has an unfilled grey tail after the green bar. Its
                # interior is darker than white digit ink and lighter than
                # the surrounding border. Require the same continuous long
                # bar, separate badge, and no excluded white glyph as above.
                gray = cv2.inRange(wide_hsv, (0, 0, 75), (180, 70, 110))
                _, gray_labels, gray_parts, _ = cv2.connectedComponentsWithStats(gray[:, 60:])
                _, bar_labels, bar_parts, _ = cv2.connectedComponentsWithStats(wide_cyan | gray)
                gray_candidates = []
                for label, part in enumerate(gray_parts[1:], 1):
                    bx, by, bw, bh, area = map(int, part)
                    if not (bx == 0 and 5 <= by <= 12 and 10 <= bh <= 20
                            and by+bh <= 26 and bw <= 14 and bw < x and 60 <= area <= 200):
                        continue
                    overlaps = set(map(int, bar_labels[:, 60:][gray_labels == label])) - {0}
                    if len(overlaps) != 1:
                        continue
                    bar_label = overlaps.pop()
                    wx, wy, ww, wh, _ = map(int, bar_parts[bar_label])
                    if not (wx == 0 and 60 < ww < 60+x and 5 <= wy <= 12
                            and 10 <= wh <= 20 and wy+wh <= 26
                            and np.array_equal(bar_labels[:, 60:] == bar_label, gray_labels == label)):
                        continue
                    # Bound the entire unfilled tail to the existing 14 px
                    # cap width, not just its portion inside the numeric ROI.
                    fills = [tuple(map(int, fill)) for fill_label, fill in enumerate(wide_parts[1:], 1)
                             if fill[0] == 0 and 10 <= fill[3] <= 20
                             and 0 < ww-fill[2] <= 14
                             and fill[1] >= wy and fill[1]+fill[3] <= wy+wh
                             and np.all(bar_labels[wide_labels == fill_label] == bar_label)]
                    if len(fills) == 1:
                        gray_candidates.append((part, [600+wx, 175+wy, ww, wh]))
                if len(gray_candidates) != 1:
                    return _hif_shield_complete_bar_roi(image, x, y, width, height)
                part, bar = gray_candidates[0]
                bx, by, bw, bh, _ = map(int, part)
                return dict(roi=[660+x, 175, 55-x, 34],
                    cyan_badge=[660+x, 175+y, width, height],
                    excluded_bar_cap=[660+bx, 175+by, bw, bh],
                    bar_cap_color='gray', connected_filled_bar=bar,
                    excluded_bar_region=[660, 175, x, 34])
            part, bar = candidates[0]
            bx, by, bw, bh, _ = map(int, part)
            return dict(roi=[660+x, 175, 55-x, 34],
                cyan_badge=[660+x, 175+y, width, height],
                excluded_bar_cap=[660+bx, 175+by, bw, bh],
                bar_cap_color='cyan', connected_filled_bar=bar)
        bx, by, bw, bh, _ = caps[0]
        return dict(roi=[660+x, 175, 55-x, 34],
            cyan_badge=[660+x, 175+y, width, height],
            excluded_bar_cap=[660+bx, 175+by, bw, bh], bar_cap_color='cyan')
    if len(relevant) != 1:
        return None
    bx, by, bw, bh, area = map(int, relevant[0])
    if bx == 0 and allow_filled_bar:
        return _hif_shield_clipped_highlight_roi(image, x, y, width, height)
    if not (bx > 0 and 0 < by <= 3 and 5 <= bw <= 14 and 24 <= bh <= 28
            and by+bh == 28 and bx+bw <= x and 60 <= area <= 200):
        return None
    return dict(roi=[660+x, 175, 55-x, 34],
        cyan_badge=[660+x, 175+y, width, height],
        excluded_bar_cap=[660+bx, 175+by, bw, bh])


def _hif_shield_clipped_highlight_roi(image, badge_x, badge_y, badge_width, badge_height):
    """Prove a white C-shaped bar end whose left edge is outside the value ROI."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image[175:203, 600:660+badge_x], cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 210), (180, 70, 255))
    _, white_labels, parts, _ = cv2.connectedComponentsWithStats(white)
    # As in the original cap proof, highlights wholly above the badge do
    # not own numeral pixels. Every component touching its band still counts.
    caps = [(i, p) for i,p in enumerate(parts[1:], 1)
            if p[1] < badge_y+badge_height and p[1]+p[3] > badge_y]
    if len(caps) != 1:
        return None
    cap_label, cap = caps[0]
    cx, cy, cw, ch, _ = map(int, cap)
    if not (0 < cx < 60 and 0 < cy <= 3 and 24 <= ch <= 28
            and cy+ch == 28 and cx+cw == 60+badge_x and cw <= ch):
        return None
    cyan = cv2.inRange(hsv, (40, 80, 150), (100, 255, 255))
    cyan[:, 60:] = cv2.inRange(hsv[:, 60:], (80, 80, 150), (100, 255, 255))
    _, labels, parts, _ = cv2.connectedComponentsWithStats(cyan)
    bodies = [(i, tuple(map(int, p))) for i,p in enumerate(parts[1:], 1)
              if p[0] == 0 and 5 <= p[1] <= 12 and 10 <= p[3] <= 20
              and p[1]+p[3] <= 26 and 60 < p[2] < 60+badge_x]
    if len(bodies) != 1:
        return None
    label, (_, by, bw, bh, _) = bodies[0]
    body, rim = labels == label, white_labels == cap_label
    # The long filled body must enter an open left side of the same cap.
    # A separate white digit, including 0, cannot supply this open boundary.
    if (cy >= by or cy+ch <= by+bh or cx >= bw-2
            or np.any(body[by:by+bh, :60].sum(axis=0) < bh-1)
            or not body[by+2:by+bh-2, cx:60].all()
            or rim[by+2:by+bh-2, cx:60].any()
            or not rim[cy:by, cx+2:bw-2].any(axis=0).all()
            or not rim[by+bh:cy+ch, cx+2:bw-2].any(axis=0).all()
            or not rim[by+2:by+bh-2, bw:cx+cw].any(axis=1).all()):
        return None
    return dict(roi=[660+badge_x, 175, 55-badge_x, 34],
        cyan_badge=[660+badge_x, 175+badge_y, badge_width, badge_height],
        excluded_bar_cap=[600+cx, 175+cy, cw, ch], bar_cap_color='white',
        connected_filled_bar=[600, 175+by, bw, bh])


def _hif_shield_complete_bar_roi(image, badge_x, badge_y, badge_width, badge_height):
    """Prove the whole capsule when its grey tail exceeds the short-cap proof."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image[170:210, 475:715], cv2.COLOR_BGR2HSV)
    if cv2.inRange(hsv[5:39, 185:185+badge_x], (0, 0, 210), (180, 70, 255)).any():
        return None
    # Split the already admitted green/cyan range into interior and rim.
    green = cv2.inRange(hsv, (40, 80, 150), (85, 255, 255))
    gray = cv2.inRange(hsv, (0, 0, 75), (180, 70, 110))
    _, labels, parts, _ = cv2.connectedComponentsWithStats(green | gray)
    bodies = [(i, tuple(map(int, part))) for i, part in enumerate(parts[1:], 1)
        if 10 <= part[0] <= 15 and 10 <= part[1] <= 17
        and 178 <= part[2] <= 185 and 10 <= part[3] <= 20
        and 191 <= part[0]+part[2] <= 195]
    if len(bodies) != 1:
        return None
    label, (x, y, width, height, _) = bodies[0]
    body = labels == label
    columns = body[y:y+height, x:x+width].sum(axis=0)
    if (np.any(columns[5:-5] < height-4) or columns.sum() < .9*width*height
            or not np.all(body[y+3:y+height-3, x+5:x+width-5], axis=1).any()
            or columns[0] > height-4 or columns[-1] > height-4):
        return None
    cyan = cv2.inRange(hsv, (86, 80, 150), (100, 255, 255))
    _, rim_labels, rims, _ = cv2.connectedComponentsWithStats(cyan)
    caps = [(i, tuple(map(int, p))) for i, p in enumerate(rims[1:], 1)
        if 0 < p[0] < x and 4 <= p[1] <= 9 and 30 <= p[2] <= 55
        and 24 <= p[3] <= 32 and p[1] < y and p[1]+p[3] > y+height]
    if len(caps) != 1:
        return None
    cap_label, (cx, cy, cw, ch, _) = caps[0]
    rim = rim_labels == cap_label
    if (not np.all(rim[cy:y, x+5:cx+cw-3].any(axis=0))
            or not np.all(rim[y+height:cy+ch, x+5:cx+cw-3].any(axis=0))
            or not np.all(rim[y:y+height, cx:x].any(axis=1))):
        return None
    # The rim proves the left end. Beyond it both dark boundaries must
    # enclose the connected interior; the right end must close separately
    # before the badge. No tail-length or remaining-stamina assumption.
    dark = cv2.inRange(hsv, (0, 0, 30), (180, 80, 80)) > 0
    for column in range(cx+cw+2, x+width-3):
        rows = np.flatnonzero(body[:, column])
        if (not len(rows) or not dark[rows[0]-3:rows[0], column].any()
                or not dark[rows[-1]+1:rows[-1]+4, column].any()):
            return None
    right = x+width
    if (475+right >= 660+badge_x
            or not np.all(dark[y+3:y+height-3, right:right+3].any(axis=1))
            or np.any(body[:, 185:] & (green[:, 185:] > 0))):
        return None
    tail = body[:, 185:]
    ty, tx = np.where(tail)
    if not len(tx) or np.any(tx >= badge_x):
        return None
    return dict(roi=[660+badge_x, 175, 55-badge_x, 34],
        cyan_badge=[660+badge_x, 175+badge_y, badge_width, badge_height],
        excluded_bar_cap=[660+int(tx.min()), 170+int(ty.min()),
                          int(tx.max()-tx.min()+1), int(ty.max()-ty.min()+1)],
        bar_cap_color='gray', connected_filled_bar=[475+x, 170+y, width, height],
        complete_bar_boundary=True, excluded_bar_region=[660, 175, badge_x, 34])


def _hif_shield_bar_owned_detection(image, roi, rows, proof):
    """Identify bar-only OCR padding without deleting any competing ink."""
    import cv2
    import numpy as np

    if not proof or not proof.get('complete_bar_boundary') or len(rows) != 2:
        return None
    if any(not _valid_ocr_result(row) or not str(getattr(row, 'text', '')).isdigit()
           or len(row.box) != 4 or any(type(v) is not int for v in row.box)
           or not (roi[0] <= row.box[0] < row.box[0]+row.box[2] <= roi[0]+roi[2]
               and roi[1] <= row.box[1] < row.box[1]+row.box[3] <= roi[1]+roi[3]) for row in rows):
        return None
    hsv = cv2.cvtColor(image[175:209, 660:715], cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 210), (180, 70, 255)) > 0
    cyan = cv2.inRange(hsv, (80, 80, 150), (100, 255, 255)) > 0
    left = proof['roi'][0]-660
    owned = []
    for row in rows:
        x, y, width, height = row.box
        x, y = x-660, y-175
        if x != 0 or np.any(white[y:y+height, x:x+width]):
            continue
        cap_x, cap_y, cap_width, cap_height = proof['excluded_bar_cap']
        if not (660+x <= cap_x and cap_x+cap_width <= 660+x+width
                and 175+y <= cap_y and cap_y+cap_height <= 175+y+height):
            continue
        for line in range(y, y+height):
            edge = np.flatnonzero(cyan[line, left:])
            # Padding may reach only the first cyan edge pixel, never the
            # badge's interior (even when that interior contains no ink).
            if x+width-1 > (left+int(edge[0]) if len(edge) else left-1):
                break
        else:
            owned.append(row)
    if len(owned) != 1:
        return None
    kept = next(row for row in rows if row is not owned[0])
    x, y, width, height = kept.box
    ink_y, ink_x = np.where(white[:, left:])
    if (not len(ink_x) or not np.all((ink_x+660+left >= x) & (ink_x+660+left < x+width)
            & (ink_y+175 >= y) & (ink_y+175 < y+height))):
        return None
    return kept


def _hif_shield_text_fragments(rows, roi):
    """Unfiltered single glyphs inside one badge, never competing full reads."""
    if not 2 <= len(rows) <= 3:
        return False
    x, y, width, height = roi
    boxes = []
    for row in rows:
        box = getattr(row, 'box', ())
        if (not _valid_ocr_result(row) or type(getattr(row, 'text', None)) is not str
                or len(row.text) != 1 or not row.text.isalnum()
                or len(box) != 4 or any(type(value) is not int for value in box)
                or not (x <= box[0] < box[0]+box[2] <= x+width
                        and y <= box[1] < box[1]+box[3] <= y+height)):
            return False
        boxes.append(box)
    boxes.sort(key=lambda box: box[0])
    return (any(row.text.isdigit() for row in rows)
        and max(box[1] for box in boxes) < min(box[1]+box[3] for box in boxes)
        and all(left[0] < right[0] and left[0]+left[2] < right[0]+right[2]
                for left, right in zip(boxes, boxes[1:])))


def _parse_live_field_ocr_results(
    results: list[Any],
    *,
    name: str,
    expected: str,
) -> dict[str, Any]:
    """Apply one field's existing regex/range semantics to OCR results."""

    semantic_results: list[tuple[Any, str, int | float]] = []
    for result in results:
        if not _valid_ocr_result(result):
            continue
        raw_text = getattr(result, "text", None)
        if type(raw_text) is not str:
            continue
        text = raw_text.strip().replace(",", "")
        if re.fullmatch(expected, text) is None:
            continue
        try:
            value: int | float = (
                int(text[:-1]) / 100.0
                if name == "score_multiplier"
                else int(text)
            )
        except ValueError:
            continue
        limits = LIVE_FIELD_LIMITS.get(name)
        if limits is None:
            continue
        lower, upper = limits
        if lower <= value <= upper:
            semantic_results.append((result, text, value))
    distinct_values = {item[2] for item in semantic_results}
    if len(distinct_values) != 1:
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": (
                "live_field_ocr_missing"
                if not semantic_results
                else "live_field_ocr_ambiguous"
            ),
            "candidate_values": sorted(distinct_values),
            "observed_result_count": len(results),
        }
    best, text, value = max(
        semantic_results,
        key=lambda item: float(getattr(item[0], "score")),
    )
    return {
        "accepted": True,
        "value": value,
        "confidence": float(getattr(best, "score", 0.0)),
        "text": text,
        "box": list(getattr(best, "box", ())),
        "observed_result_count": len(results),
    }


def _candidate_relative_roi(
    candidate: Any,
    relative_roi: tuple[int, int, int, int] = (
        LIVE_SELECTED_EFFECTIVE_COST_RELATIVE_ROI
    ),
) -> list[int]:
    """Place a fixed canonical offset relative to one detected card candidate."""

    box = getattr(candidate, "box", None)
    if (
        not isinstance(box, (list, tuple))
        or len(box) != 4
        or any(type(value) is not int for value in box)
        or box[2] <= 0
        or box[3] <= 0
    ):
        raise ValueError("invalid_live_candidate_box")
    if (
        not isinstance(relative_roi, tuple)
        or len(relative_roi) != 4
        or any(type(value) is not int for value in relative_roi)
    ):
        raise ValueError("invalid_live_candidate_relative_roi")
    relative_x, relative_y, relative_width, relative_height = relative_roi
    x, y, _width, _height = box
    roi = [
        x + relative_x,
        y + relative_y,
        relative_width,
        relative_height,
    ]
    if roi[0] < 0 or roi[1] < 0:
        raise ValueError("live_candidate_relative_roi_outside_frame")
    return roi


def _valid_ocr_result(result: Any) -> bool:
    raw_box = getattr(result, "box", ())
    raw_score = getattr(result, "score", None)
    return bool(
        isinstance(raw_box, (list, tuple))
        and len(raw_box) == 4
        and all(type(value) is int for value in raw_box)
        and raw_box[2] > 0
        and raw_box[3] > 0
        and not isinstance(raw_score, bool)
        and isinstance(raw_score, (int, float))
        and math.isfinite(float(raw_score))
        and 0.0 <= float(raw_score) <= 1.0
    )


def _live_candidate_signature(candidates: tuple[Any, ...]) -> tuple[Any, ...]:
    """Freeze the exact candidate geometry used to build one OCR mosaic."""

    signature = []
    for candidate in candidates:
        box = getattr(candidate, "box", None)
        slot = getattr(candidate, "slot", None)
        if (
            type(slot) is not int
            or not isinstance(box, (list, tuple))
            or len(box) != 4
            or any(type(value) is not int for value in box)
        ):
            raise ValueError("authoritative_live_candidate_signature_invalid")
        signature.append((slot, *box))
    return tuple(signature)


def _build_fixed_live_ocr_mosaic(
    sources: list[dict[str, Any]],
    *,
    cell_size: tuple[int, int] = LIVE_AUTHORITATIVE_OCR_MOSAIC_CELL,
    mosaic_shape: tuple[int, int] = LIVE_AUTHORITATIVE_OCR_MOSAIC_SHAPE,
) -> tuple[Any | None, dict[str, dict[str, Any]], str | None]:
    """Copy bounded source ROIs into fixed 240x100 OCR cells without scaling."""

    try:
        import numpy as np

        mosaic_height, mosaic_width = mosaic_shape
        cell_width, cell_height = cell_size
        first_source = np.asarray(sources[0]["image"])
        if first_source.ndim != 3 or first_source.shape[2] < 3:
            raise ValueError("authoritative_live_mosaic_source_unavailable")
        mosaic = np.full(
            (mosaic_height, mosaic_width, first_source.shape[2]),
            255,
            dtype=first_source.dtype,
        )
        placements: dict[str, dict[str, Any]] = {}
        occupied_cells: set[tuple[int, int]] = set()
        for source in sources:
            key = source.get("key")
            row = source.get("row")
            column = source.get("column")
            roi = source.get("roi")
            source_image = np.asarray(source.get("image"))
            if (
                type(key) is not str
                or not key
                or type(row) is not int
                or type(column) is not int
                or not 0 <= row < mosaic_height // cell_height
                or not 0 <= column < mosaic_width // cell_width
                or (row, column) in occupied_cells
                or not isinstance(roi, (list, tuple))
                or len(roi) != 4
                or any(type(value) is not int for value in roi)
                or source_image.ndim != 3
                or source_image.shape[2] != mosaic.shape[2]
            ):
                raise ValueError("authoritative_live_mosaic_binding_invalid")
            x, y, width, height = roi
            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or width > cell_width
                or height > cell_height
                or source_image.shape[0] <= y
                or source_image.shape[1] <= x
            ):
                raise ValueError("authoritative_live_mosaic_source_roi_invalid")
            cell_x = column * cell_width + (cell_width - width) // 2
            cell_y = row * cell_height + (cell_height - height) // 2
            crop = np.full(
                (height, width, source_image.shape[2]),
                255,
                dtype=source_image.dtype,
            )
            available_height = min(height, source_image.shape[0] - y)
            available_width = min(width, source_image.shape[1] - x)
            crop[:available_height, :available_width] = source_image[
                y : y + available_height,
                x : x + available_width,
            ]
            mosaic[cell_y : cell_y + height, cell_x : cell_x + width] = crop
            placements[key] = {
                "key": key,
                "source_roi": [x, y, width, height],
                "mosaic_roi": [cell_x, cell_y, width, height],
                "cell_roi": [
                    column * cell_width,
                    row * cell_height,
                    cell_width,
                    cell_height,
                ],
                "cell": [column, row],
                "expected": source.get("expected"),
            }
            occupied_cells.add((row, column))
        return mosaic, placements, None
    except (AttributeError, ImportError, IndexError, TypeError, ValueError):
        return None, {}, "authoritative_live_mosaic_source_unavailable"


def _translate_mosaic_ocr_result(
    result: Any,
    placement: dict[str, Any],
) -> Any:
    mosaic_x, mosaic_y, _mosaic_width, _mosaic_height = placement["mosaic_roi"]
    source_x, source_y, _source_width, _source_height = placement["source_roi"]
    box_x, box_y, box_width, box_height = getattr(result, "box")
    return SimpleNamespace(
        text=getattr(result, "text", None),
        score=getattr(result, "score", None),
        box=[
            source_x + box_x - mosaic_x,
            source_y + box_y - mosaic_y,
            box_width,
            box_height,
        ],
    )


def _recognize_fixed_live_ocr_mosaic(
    context: Context,
    mosaic: Any,
    placements: dict[str, dict[str, Any]],
    *,
    node_name: str,
    cell_size: tuple[int, int] = LIVE_AUTHORITATIVE_OCR_MOSAIC_CELL,
    mosaic_shape: tuple[int, int] = LIVE_AUTHORITATIVE_OCR_MOSAIC_SHAPE,
) -> dict[str, Any]:
    """Run one OCR and strictly bind every result to exactly one populated cell."""

    started_at = time.perf_counter_ns()
    grouped: dict[str, list[Any]] = {key: [] for key in placements}

    def outcome(
        *,
        supported: bool,
        accepted: bool,
        reason: str,
        contradictions: list[dict[str, Any]],
        observed_result_count: int = 0,
        ignored_result_count: int = 0,
    ) -> dict[str, Any]:
        return {
            "supported": supported,
            "accepted": accepted,
            "reason": reason,
            "results": grouped,
            "contradictions": contradictions,
            "observed_result_count": observed_result_count,
            "ignored_result_count": ignored_result_count,
            "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
        }

    pipeline_override = {
        node_name: {
            "recognition": "OCR",
            "roi": [0, 0, mosaic_shape[1], mosaic_shape[0]],
            "expected": r"^(?:SELECT|\d{1,9}%?|[+\-−]\d{1,3})$",
        }
    }
    try:
        recognizer = getattr(context, "run_recognition", None)
    except Exception:
        return outcome(
            supported=True,
            accepted=False,
            reason="authoritative_live_mosaic_contradiction",
            contradictions=[{"reason": "ocr_context_api_structure_invalid"}],
        )
    if not callable(recognizer):
        return outcome(
            supported=False,
            accepted=False,
            reason="authoritative_live_mosaic_api_unavailable",
            contradictions=[],
        )
    try:
        recognizer_signature = inspect.signature(recognizer)
    except (TypeError, ValueError):
        recognizer_signature = None
    if recognizer_signature is not None:
        try:
            recognizer_signature.bind(
                node_name,
                mosaic,
                pipeline_override=pipeline_override,
            )
        except TypeError:
            return outcome(
                supported=False,
                accepted=False,
                reason="authoritative_live_mosaic_api_unavailable",
                contradictions=[],
            )
    try:
        detail = recognizer(
            node_name,
            mosaic,
            pipeline_override=pipeline_override,
        )
    except TypeError:
        return outcome(
            supported=True,
            accepted=False,
            reason="authoritative_live_mosaic_contradiction",
            contradictions=[{"reason": "ocr_context_api_type_error"}],
        )

    if detail is None:
        return outcome(
            supported=True,
            accepted=False,
            reason="authoritative_live_mosaic_contradiction",
            contradictions=[{"reason": "ocr_detail_missing"}],
        )
    try:
        hit = getattr(detail, "hit")
        raw_all_results = getattr(detail, "all_results")
        raw_filtered_results = getattr(detail, "filtered_results")
    except Exception:
        return outcome(
            supported=True,
            accepted=False,
            reason="authoritative_live_mosaic_contradiction",
            contradictions=[{"reason": "ocr_detail_structure_invalid"}],
        )
    if (
        type(hit) is not bool
        or not isinstance(raw_all_results, (list, tuple))
        or not isinstance(raw_filtered_results, (list, tuple))
    ):
        return outcome(
            supported=True,
            accepted=False,
            reason="authoritative_live_mosaic_contradiction",
            contradictions=[{"reason": "ocr_detail_structure_invalid"}],
        )

    def normalize_results(
        raw_results: list[Any] | tuple[Any, ...],
        *,
        collection: str,
    ) -> tuple[list[Any] | None, dict[str, Any] | None]:
        normalized: list[Any] = []
        for index, result in enumerate(raw_results):
            try:
                raw_text = getattr(result, "text")
                raw_score = getattr(result, "score")
                raw_box = getattr(result, "box")
            except Exception:
                return None, {
                    "reason": "ocr_result_structure_invalid",
                    "collection": collection,
                    "index": index,
                }
            if type(raw_text) is not str:
                return None, {
                    "reason": "ocr_result_text_not_string",
                    "collection": collection,
                    "index": index,
                }
            if (
                type(raw_score) not in {int, float}
                or not math.isfinite(float(raw_score))
                or not 0.0 <= float(raw_score) <= 1.0
            ):
                return None, {
                    "reason": "ocr_result_score_invalid",
                    "collection": collection,
                    "index": index,
                }
            if (
                not isinstance(raw_box, (list, tuple))
                or len(raw_box) != 4
                or any(type(value) is not int for value in raw_box)
                or raw_box[2] <= 0
                or raw_box[3] <= 0
            ):
                return None, {
                    "reason": "ocr_result_box_invalid",
                    "collection": collection,
                    "index": index,
                }
            normalized.append(
                SimpleNamespace(
                    text=raw_text,
                    score=float(raw_score),
                    box=list(raw_box),
                )
            )
        return normalized, None

    all_results, all_failure = normalize_results(
        raw_all_results,
        collection="all_results",
    )
    if all_failure is not None:
        return outcome(
            supported=True,
            accepted=False,
            reason="authoritative_live_mosaic_contradiction",
            contradictions=[all_failure],
        )
    filtered_results, filtered_failure = normalize_results(
        raw_filtered_results,
        collection="filtered_results",
    )
    if filtered_failure is not None:
        return outcome(
            supported=True,
            accepted=False,
            reason="authoritative_live_mosaic_contradiction",
            contradictions=[filtered_failure],
        )
    assert all_results is not None
    assert filtered_results is not None
    if hit is not True:
        all_results = []
        filtered_results = []
    observed_results = all_results or filtered_results
    contradictions: list[dict[str, Any]] = []
    ignored_result_count = 0
    legal_union = re.compile(r"^(?:SELECT|\d{1,9}%?|[+\-−]\d{1,3})$")
    occupied_cells = {
        tuple(placement["cell"]) for placement in placements.values()
    }
    filtered_signatures = {
        (
            result.text,
            result.score,
            tuple(result.box),
        )
        for result in filtered_results
    }

    def box_bound_to_placement(
        placement: dict[str, Any],
        box: tuple[int, int, int, int] | list[int],
    ) -> bool:
        """Allow only the calibrated one-pixel OCR pad into an empty neighbour."""

        box_x, box_y, box_width, box_height = box
        cell_x, cell_y, cell_width, cell_height = placement["cell_roi"]
        column, row = placement["cell"]
        cell_right = cell_x + cell_width
        cell_bottom = cell_y + cell_height
        row_count = mosaic_shape[0] // cell_height
        left_ok = box_x > cell_x
        right = box_x + box_width
        right_ok = right < cell_right
        top_ok = box_y > cell_y
        bottom = box_y + box_height
        bottom_ok = bottom < cell_bottom or (
            placement["key"] == "turns_remaining"
            and placement["cell"] == [2, 2]
            and bottom == cell_bottom + 1
            and row + 1 < row_count
            and (column, row + 1) not in occupied_cells
        )
        mosaic_x, mosaic_y, mosaic_width, mosaic_height = placement["mosaic_roi"]
        center_x = box_x + box_width / 2.0
        center_y = box_y + box_height / 2.0
        return bool(
            left_ok
            and right_ok
            and top_ok
            and bottom_ok
            and mosaic_x <= center_x < mosaic_x + mosaic_width
            and mosaic_y <= center_y < mosaic_y + mosaic_height
        )

    for result in observed_results:
        raw_text = getattr(result, "text", None)
        if type(raw_text) is not str:
            contradictions.append({"reason": "ocr_result_text_not_string"})
            continue
        text = raw_text.strip().replace(",", "")
        if legal_union.fullmatch(text) is None:
            ignored_result_count += 1
            continue
        if not _valid_ocr_result(result):
            contradictions.append(
                {
                    "reason": "legal_ocr_result_structure_invalid",
                    "text": text,
                }
            )
            continue
        box_x, box_y, box_width, box_height = getattr(result, "box")
        bound = [
            placement
            for placement in placements.values()
            if box_bound_to_placement(placement, getattr(result, "box"))
        ]
        if len(bound) != 1:
            cell_width, cell_height = cell_size
            left_cell = box_x // cell_width
            right_cell = (box_x + box_width - 1) // cell_width
            top_cell = box_y // cell_height
            bottom_cell = (box_y + box_height - 1) // cell_height
            if left_cell != right_cell or top_cell != bottom_cell:
                contradictions.append(
                    {
                        "reason": "ocr_result_crosses_mosaic_cell",
                        "box": list(getattr(result, "box")),
                        "text": getattr(result, "text"),
                    }
                )
            else:
                cell_placements = [
                    placement
                    for placement in placements.values()
                    if placement["cell"] == [left_cell, top_cell]
                ]
                if cell_placements:
                    contradictions.append(
                        {
                            "reason": "ocr_result_misses_source_binding",
                            "key": cell_placements[0]["key"],
                            "box": list(getattr(result, "box")),
                            "text": getattr(result, "text"),
                        }
                    )
                else:
                    ignored_result_count += 1
            continue
        placement = bound[0]
        expected = placement.get("expected")
        if type(expected) is not str or re.fullmatch(expected, text) is None:
            contradictions.append(
                {
                    "reason": "ocr_result_wrong_tile_semantics",
                    "key": placement["key"],
                    "text": text,
                    "box": list(getattr(result, "box")),
                }
            )
            continue
        result_signature = (
            getattr(result, "text", None),
            getattr(result, "score", None),
            tuple(getattr(result, "box", ())),
        )
        if (
            placement["key"]
            not in {"stamina", "shield", "shield_numeric_badge", "stamina_delta", "shield_delta"}
            and result_signature not in filtered_signatures
        ):
            ignored_result_count += 1
            continue
        grouped[placement["key"]].append(
            _translate_mosaic_ocr_result(result, placement)
        )
    for key, results in grouped.items():
        if len(results) > 1:
            contradictions.append(
                {
                    "reason": "multiple_legal_results_in_mosaic_tile",
                    "key": key,
                    "texts": [getattr(result, "text", None) for result in results],
                }
            )
    return outcome(
        supported=True,
        accepted=not contradictions,
        reason=(
            "authoritative_live_mosaic_bound"
            if not contradictions
            else "authoritative_live_mosaic_contradiction"
        ),
        contradictions=contradictions,
        observed_result_count=len(observed_results),
        ignored_result_count=ignored_result_count,
    )


def _parse_live_score_zero_results(
    results: list[Any],
    *,
    roi: list[int],
) -> dict[str, Any]:
    """Apply the dedicated zero-glyph confidence and box gates to mosaic OCR."""

    if len(results) != 1 or not _valid_ocr_result(results[0]):
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": (
                "self_score_zero_missing"
                if not results
                else "self_score_zero_ambiguous"
            ),
        }
    result = results[0]
    text = str(getattr(result, "text", "")).strip()
    confidence = float(getattr(result, "score"))
    box = list(getattr(result, "box"))
    roi_x, roi_y, roi_width, roi_height = roi
    box_x, box_y, box_width, box_height = box
    min_width, max_width = LIVE_SELF_SCORE_ZERO_BOX_WIDTH_RANGE
    min_height, max_height = LIVE_SELF_SCORE_ZERO_BOX_HEIGHT_RANGE
    if not (
        text == "0"
        and confidence >= LIVE_SELF_SCORE_ZERO_MIN_CONFIDENCE
        and roi_x <= box_x
        and roi_y <= box_y
        and box_x + box_width <= roi_x + roi_width
        and box_y + box_height <= roi_y + roi_height
        and min_width <= box_width <= max_width
        and min_height <= box_height <= max_height
    ):
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": "self_score_zero_semantics_rejected",
            "text": text,
            "box": box,
        }
    return {
        "accepted": True,
        "value": 0,
        "confidence": confidence,
        "reason": "self_score_zero_mosaic_dedicated_roi",
        "text": text,
        "box": box,
        "roi": list(roi),
    }


def _ocr_live_score(
    context: Context,
    image: Any,
    action_order: dict[str, Any],
    *,
    observed_wide_miss: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read self score, with a geometry-bound fallback for an exact zero glyph."""

    if not isinstance(action_order, dict) or action_order.get("accepted") is not True:
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": "self_score_action_order_unresolved",
        }
    score_roi = action_order.get("score_roi")
    if (
        not isinstance(score_roi, (list, tuple))
        or len(score_roi) != 4
        or any(type(value) is not int for value in score_roi)
        or score_roi[0] < 0
        or score_roi[1] < 0
        or score_roi[2] <= 0
        or score_roi[3] <= 0
    ):
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": "self_score_wide_roi_invalid",
        }
    if (isinstance(observed_wide_miss, dict)
            and observed_wide_miss.get('accepted') is False
            and observed_wide_miss.get('reason') == 'live_field_ocr_missing'
            and observed_wide_miss.get('value') is None
            and type(observed_wide_miss.get('observed_result_count')) is int
            and observed_wide_miss['observed_result_count'] == 0):
        # The caller has already read this frame's same wide ROI in its
        # common mosaic. Only an explicit empty result can reuse that work.
        wide_result = dict(observed_wide_miss)
    else:
        wide_result = _ocr_field(
            context,
            image,
            "score",
            list(score_roi),
            r"^\d+$",
        )
    if wide_result.get("accepted") is True and (
        action_order.get("layout") not in (
            "hif-three-player-opening-v1", "hif-four-player-fourth-v1")
        or wide_result.get("confidence", 0.0) >= LIVE_SELF_SCORE_ZERO_MIN_CONFIDENCE
    ):
        return wide_result

    anchor_x = action_order.get("anchor_x")
    relative_x, y, width, height = LIVE_SELF_SCORE_ZERO_ROI_OFFSET
    zero_roi = (
        [anchor_x + relative_x, y, width, height]
        if type(anchor_x) is int
        else None
    )
    if action_order.get("layout") == "hif-three-player-opening-v1" and anchor_x == 444:
        # HIF's zero is smaller and farther left than the NIA expanded column.
        zero_roi = [366, 126, 34, 40]
    elif action_order.get("layout") == "hif-four-player-fourth-v1" and anchor_x == 514:
        # Native 1716: independent zero glyph at [436,131,28,32], .99051.
        # Keep the dedicated strict zero check; the wide row read was OAY.
        zero_roi = [436, 126, 34, 40]
    raw_shape = getattr(image, "shape", None)
    shape = tuple(raw_shape) if isinstance(raw_shape, (list, tuple)) else ()
    if (
        zero_roi is None
        or zero_roi[0] < 0
        or len(shape) < 2
        or any(type(value) is not int or value <= 0 for value in shape[:2])
        or shape[0] < zero_roi[1] + zero_roi[3]
        or shape[1] < zero_roi[0] + zero_roi[2]
    ):
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": "self_score_zero_roi_invalid",
            "wide_result": wide_result,
            "roi": zero_roi,
        }

    node_name = "Task250VisualSelfScoreZero"
    detail = context.run_recognition(
        node_name,
        image,
        pipeline_override={
            node_name: {
                "recognition": "OCR",
                "roi": zero_roi,
                "expected": r"^0$",
            }
        },
    )
    all_results = (
        list(getattr(detail, "all_results", ()) or ())
        if detail and detail.hit
        else []
    )
    filtered_results = (
        list(getattr(detail, "filtered_results", ()) or ())
        if detail and detail.hit
        else []
    )
    if len(all_results) != 1 or len(filtered_results) != 1:
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": (
                "self_score_zero_missing"
                if not all_results or not filtered_results
                else "self_score_zero_ambiguous"
            ),
            "wide_result": wide_result,
            "roi": zero_roi,
            "observed_all_result_count": len(all_results),
            "observed_filtered_result_count": len(filtered_results),
        }

    result = all_results[0]
    filtered_result = filtered_results[0]
    raw_text = getattr(result, "text", None)
    raw_filtered_text = getattr(filtered_result, "text", None)
    if not (
        _valid_ocr_result(result)
        and _valid_ocr_result(filtered_result)
        and type(raw_text) is str
        and type(raw_filtered_text) is str
    ):
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": "self_score_zero_result_invalid",
            "wide_result": wide_result,
            "roi": zero_roi,
            "observed_all_result_count": 1,
            "observed_filtered_result_count": 1,
        }
    result_box = list(getattr(result, "box"))
    filtered_box = list(getattr(filtered_result, "box"))
    result_confidence = float(getattr(result, "score"))
    filtered_confidence = float(getattr(filtered_result, "score"))
    if not (
        raw_text.strip() == "0"
        and raw_filtered_text.strip() == "0"
        and result_confidence == filtered_confidence
        and result_box == filtered_box
    ):
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": "self_score_zero_result_mismatch",
            "wide_result": wide_result,
            "roi": zero_roi,
            "observed_all_result_count": 1,
            "observed_filtered_result_count": 1,
        }
    text = raw_text.strip()
    confidence = result_confidence
    box = result_box
    roi_x, roi_y, roi_width, roi_height = zero_roi
    box_x, box_y, box_width, box_height = box
    min_width, max_width = LIVE_SELF_SCORE_ZERO_BOX_WIDTH_RANGE
    min_height, max_height = LIVE_SELF_SCORE_ZERO_BOX_HEIGHT_RANGE
    box_contained = (
        roi_x <= box_x
        and roi_y <= box_y
        and box_x + box_width <= roi_x + roi_width
        and box_y + box_height <= roi_y + roi_height
    )
    if not (
        text == "0"
        and confidence >= LIVE_SELF_SCORE_ZERO_MIN_CONFIDENCE
        and box_contained
        and min_width <= box_width <= max_width
        and min_height <= box_height <= max_height
    ):
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": "self_score_zero_semantics_rejected",
            "wide_result": wide_result,
            "text": text,
            "box": box,
            "roi": zero_roi,
            "observed_all_result_count": 1,
            "observed_filtered_result_count": 1,
        }
    return {
        "accepted": True,
        "value": 0,
        "confidence": confidence,
        "reason": "self_score_zero_dedicated_roi",
        "text": text,
        "box": box,
        "roi": zero_roi,
        "wide_result": wide_result,
        "observed_all_result_count": 1,
        "observed_filtered_result_count": 1,
    }


class _SelectedCostOcrContext:
    """Reuse only identical OCR work inside one selected three-frame refresh."""

    def __init__(self, context):
        self._context = context
        self._observations = []
        self.counts = dict(requests=0, computations=0, reuses=0)

    def __getattr__(self, name):
        return getattr(self._context, name)

    def run_recognition(self, node, image, *, pipeline_override=None, **kwargs):
        from copy import deepcopy

        import numpy as np

        self.counts['requests'] += 1
        params = (pipeline_override.get(node) if isinstance(pipeline_override, dict)
                  and set(pipeline_override) == {node} else None)
        pixels = None
        if (not kwargs and isinstance(params, dict) and params.get('recognition') == 'OCR'
                and isinstance(image, np.ndarray) and image.ndim == 3 and image.size
                and (params.get('only_rec') is True
                     # This fixed reader consumes the entire signed-fee crop
                     # with detection enabled. Its exact input and override,
                     # not just recognized text, must match within this refresh.
                     or ((node == 'HifSkillSupportDetailCost'
                          and image.shape in ((32,40,3),(44,56,3)))
                         or (node in ('HifSkillSupportDetailTitle','HifSkillSupportDetailTitleText')
                             and image.shape == (44,446,3)))
                     and image.dtype == np.uint8
                     and params == dict(recognition='OCR',expected='.+'))):
            roi = params.get('roi', (0, 0, image.shape[1], image.shape[0]))
            if (isinstance(roi, (tuple, list)) and len(roi) == 4
                    and all(type(value) is int for value in roi)):
                x,y,w,h = roi
                if (x >= 0 and y >= 0 and w > 0 and h > 0
                        and x+w <= image.shape[1] and y+h <= image.shape[0]):
                    pixels = image[y:y+h,x:x+w]
        if pixels is not None:
            for old_node, override, previous, result in self._observations:
                if (node == old_node and pipeline_override == override
                        and pixels.shape == previous.shape and pixels.dtype == previous.dtype
                        and np.array_equal(pixels, previous)):
                    self.counts['reuses'] += 1
                    return deepcopy(result)
        self.counts['computations'] += 1
        detail = self._context.run_recognition(node, image,
            pipeline_override=pipeline_override, **kwargs)
        if (pixels is not None and type(getattr(detail, 'hit', None)) is bool
                and isinstance(getattr(detail, 'filtered_results', None), (list, tuple))):
            # Maa OCR rows are mutable dataclasses. Retain detached text/box/
            # confidence results, never RPC IDs, debug images or input evidence.
            result = SimpleNamespace(hit=detail.hit,filtered_results=deepcopy(detail.filtered_results))
            self._observations.append((node,deepcopy(pipeline_override),pixels.copy(),result))
        return detail


def _ocr_selected_effective_cost(
    context: Context,
    image: Any,
    candidate: Any,
) -> dict[str, Any]:
    """Read the selected card-face cost; HUD red deltas are never card cost."""

    try:
        roi = _candidate_relative_roi(candidate)
        if (calibrated_live_client_size(getattr(context, 'gkh_hif_client_size', None)) == (480, 853)
                or isinstance(getattr(context, 'gkh_hif_live_context', None), HifLiveRunContext)):
            # Native downscaling leaves the heart tip and nearby decorations
            # in the old crop. Keep the complete signed numeral inside it.
            roi = [roi[0]+5, roi[1]+5, 40, 32]
        shape = tuple(getattr(image, "shape", ()))
        if (
            len(shape) < 2
            or shape[0] < roi[1] + roi[3]
            or shape[1] < roi[0] + roi[2]
        ):
            raise ValueError("selected_effective_cost_roi_outside_frame")
    except (AttributeError, TypeError, ValueError):
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": "selected_effective_cost_roi_invalid",
            "roi": None,
        }

    node_name = "Task250SelectedEffectiveCost"
    detail = context.run_recognition(
        node_name,
        image,
        pipeline_override={
            node_name: {
                "recognition": "OCR",
                "roi": roi,
                "expected": r"^(?:0|[\-−]\d{1,3})$",
                "only_rec": True,
            }
        },
    )
    results = (
        list(getattr(detail, "filtered_results", ()) or ())
        if detail and detail.hit
        else []
    )
    roi_x, roi_y, roi_width, roi_height = roi
    card_face_readings = list(results)
    white_cost_observation = None
    weak_single = (len(results)==1 and _valid_ocr_result(results[0]) and results[0].score < .95)
    zero_probe_rows = None
    # A literal low-confidence zero has its own corroborating crop. Try it
    # before the two generic text transforms, which both fail on 1071's zero.
    if (weak_single and results[0].text.strip() == '0'
            and roi_x <= results[0].box[0] + results[0].box[2]/2 < roi_x + roi_width
            and roi_y <= results[0].box[1] + results[0].box[3]/2 < roi_y + roi_height
            and getattr(context, 'gkh_hif_opening_state', None) is not None):
        node = 'HifSelectedZeroCost'
        area = [roi[0]+6, roi[1]+2, 30, 34]
        detail = context.run_recognition(node, image, pipeline_override={node: dict(
            recognition='OCR', roi=area, only_rec=True, expected='.+')})
        rows = list(getattr(detail, 'filtered_results', ()) or ()) if detail and detail.hit else []
        zero_probe_rows = rows
        if (len(rows) == 1 and _valid_ocr_result(rows[0]) and rows[0].text.strip() == '0'
                and .95 <= rows[0].score <= 1):
            return dict(accepted=True, value=0, confidence=float(rows[0].score),
                reason='selected_effective_cost_card_face', text='0', box=list(results[0].box),
                roi=roi, observed_result_count=1, corroboration=dict(
                    source='selected_zero_cost_card_face', card_face_confidence=float(results[0].score),
                    roi=area, text=rows[0].text))
        if (len(rows) == 1 and _valid_ocr_result(rows[0]) and rows[0].text.strip() == '0'
                and isinstance(getattr(context, 'gkh_hif_live_context', None), HifLiveRunContext)
                and calibrated_live_client_size(getattr(context, 'gkh_hif_client_size', None)) == (480, 853)):
            from live_decision.hif_skill_support import read_complete_zero_cost

            # Both original crops read zero. Reuse the existing whole-glyph
            # reader before generic transforms; it rejects a sign or cut edge
            # and still requires an independent .95 OCR result.
            zero = read_complete_zero_cost(context, image, roi)
            if zero is not None:
                return dict(accepted=True, value=0, confidence=zero['confidence'],
                    reason='selected_effective_cost_card_face', text='0', box=list(results[0].box),
                    roi=roi, observed_result_count=1, corroboration=zero)
    if (not results or weak_single) and isinstance(getattr(context, 'gkh_hif_live_context', None), HifLiveRunContext):
        from live_decision.inventory import _inventory_ocr, inventory_white_text
        x,y,w,h=roi
        white_rows=_inventory_ocr(context,inventory_white_text(image[y:y+h,x:x+w].copy()),
                                  'HifSelectedCostWhite',only_rec=True)
        card_face_readings.extend(white_rows)
        if (len(white_rows)==1 and _valid_ocr_result(white_rows[0])
                and .95<=white_rows[0].score<=1
                and re.fullmatch(r'(?:0|[-−]\d{1,3})',white_rows[0].text.strip())
                and (not results or results[0].text.strip().replace('−','-')
                     == white_rows[0].text.strip().replace('−','-'))):
            row=white_rows[0]
            results=[SimpleNamespace(text=row.text,score=row.score,box=list(roi))]
            white_cost_observation=dict(source='selected_card_face_white_text',
                text=row.text,confidence=row.score,roi=list(roi))
        if (white_cost_observation is None and weak_single and results[0].text.strip() == '0'
                and zero_probe_rows is not None and len(zero_probe_rows)==1
                and _valid_ocr_result(zero_probe_rows[0]) and zero_probe_rows[0].text.strip()=='0'
                and len(white_rows)==1 and _valid_ocr_result(white_rows[0]) and white_rows[0].text.strip()=='0'
                and getattr(context, 'gkh_hif_client_size', None) in ((540,960),(541,961),(541,962),(542,962))):
            # 1365's resampled zero retains grey glyph edges that the binary
            # white mask loses. Reuse the fixed continuous contrast projection
            # on the complete signed-cost ROI; only an independent same zero
            # above the unchanged confidence gate can corroborate the raw read.
            import cv2
            import numpy as np

            gray=np.clip((image[y:y+h,x:x+w].min(2).astype(np.float32)-150)*(255/105),
                         0,255).astype(np.uint8)
            continuous=cv2.cvtColor(cv2.copyMakeBorder(255-gray,2,2,2,2,
                cv2.BORDER_CONSTANT,value=255),cv2.COLOR_GRAY2BGR)
            rows=_inventory_ocr(context,continuous,'HifSelectedCostContinuous',only_rec=True)
            if any(_valid_ocr_result(row) and re.fullmatch(r'[-−]?\d{1,3}',row.text.strip())
                    and int(row.text.strip().replace('−','-')) != 0 for row in rows):
                return dict(accepted=False,value=None,confidence=0.,
                    reason='selected_effective_cost_continuous_contradiction',roi=list(roi),
                    raw_text=results[0].text,observed_result_count=len(rows),
                    continuous_results=[dict(text=row.text,confidence=row.score) for row in rows])
            if (len(rows)==1 and _valid_ocr_result(rows[0]) and rows[0].text.strip()=='0'
                    and .95<=rows[0].score<=1):
                row=rows[0]
                results=[SimpleNamespace(text=row.text,score=row.score,box=list(roi))]
                white_cost_observation=dict(source='selected_card_face_continuous_text',
                    text=row.text,confidence=row.score,roi=list(roi))
        if white_cost_observation is None:
            tight_roi=[x,y+5,w,h-5]
            tight_rows=_inventory_ocr(context,image[y+5:y+h,x:x+w].copy(),
                                      'HifSelectedCostText',only_rec=True)
            card_face_readings.extend(tight_rows)
            if (len(tight_rows)==1 and _valid_ocr_result(tight_rows[0])
                    and .95<=tight_rows[0].score<=1
                    and re.fullmatch(r'(?:0|[-−]\d{1,3})',tight_rows[0].text.strip())
                    and (not results or results[0].text.strip().replace('−','-')
                         == tight_rows[0].text.strip().replace('−','-'))):
                row=tight_rows[0]
                results=[SimpleNamespace(text=row.text,score=row.score,box=tight_roi)]
                white_cost_observation=dict(source='selected_card_face_text_region',
                    text=row.text,confidence=row.score,roi=tight_roi)
            bundle=getattr(context,'gkh_hif_opening_state',None)
            if (white_cost_observation is None and not results and bundle is not None
                    and 0<=candidate.slot<len(bundle.state.deck.hand)
                    and len(white_rows)==len(tight_rows)==1
                    and all(_valid_ocr_result(row) and row.text.strip()=='0'
                            for row in (*white_rows,*tight_rows))
                    and not any(_valid_ocr_result(row)
                        and re.fullmatch(r'[-−]?\d{1,3}',row.text.strip())
                        and int(row.text.strip().replace('−','-')) != 0
                        for row in (getattr(detail,'all_results',()) or ()))):
                from live_decision.hif_skill_support import read_complete_zero_cost

                # Two existing full-cost crops agree on zero but are weak.
                # The complete enclosed glyph can supply an independent .95
                # read; a minus, cropped digit or side ink cannot be removed.
                zero = read_complete_zero_cost(context,image,roi)
                if zero is not None:
                    results=[SimpleNamespace(text='0',score=zero['confidence'],box=list(roi))]
                    white_cost_observation=zero
            if (white_cost_observation is None and (not results or weak_single) and bundle is not None
                    and 0<=candidate.slot<len(bundle.state.deck.hand)):
                from live_decision.high_score.kernel import CardCostKind
                definition=bundle.kernel.cards[bundle.state.deck.hand[candidate.slot].card_id]
                if definition.cost_kind in (CardCostKind.CONCENTRATION, CardCostKind.GOOD_CONDITION):
                    import cv2
                    import numpy as np
                    from live_decision.hif_skill_support import _read_printed_cost

                    # 1382's blue icon crosses the complete signed-fee ROI's
                    # upper edge. Reuse enclosed lettering only when every
                    # removed component is above all retained glyphs; a cut
                    # numeral or minus at the sides/bottom must remain unknown.
                    _,_,stats,_=cv2.connectedComponentsWithStats(
                        (image[y:y+h,x:x+w].min(axis=2)>=220).astype(np.uint8))
                    enclosed=[]
                    boundary=[]
                    for sx,sy,sw,sh,_area in stats[1:]:
                        (enclosed if sx>0 and sy>0 and sx+sw<w and sy+sh<h
                         else boundary).append((int(sx),int(sy),int(sw),int(sh)))
                    zero_glyph = (len(enclosed)==1 and weak_single and results[0].text.strip()=='0'
                        and enclosed[0][2]>=7 and enclosed[0][3]>=12)
                    if ((len(enclosed)>=2 or zero_glyph) and boundary
                            and max(sy+sh for _,sy,_,sh in boundary)
                            <=min(sy for _,sy,_,_ in enclosed)):
                        rows,raw,source=_read_printed_cost(context,image,roi,require_signed=not zero_glyph)
                        card_face_readings.extend(SimpleNamespace(text=row['text'],
                            score=row['confidence'],box=row['box']) for row in raw)
                        if (source=='enclosed_white_text' and len(rows)==1
                                and rows[0]['box']==roi and .95<=rows[0]['confidence']<=1
                                and re.fullmatch(r'0' if zero_glyph else r'[-−][1-9]\d{0,2}',
                                                 rows[0]['text'].strip())):
                            value=abs(int(rows[0]['text'].strip().replace('−','-')))
                            previous=[row.text.strip().replace('−','-')
                                for row in (*results,*white_rows,*tight_rows) if _valid_ocr_result(row)]
                            if any(re.fullmatch(r'[-−]?\d{1,3}',text)
                                    and abs(int(text))!=value for text in previous):
                                return dict(accepted=False,value=None,confidence=0.,
                                    reason='selected_effective_cost_enclosed_contradiction',
                                    roi=list(roi),prior_text=previous,enclosed_results=rows)
                            row=rows[0]
                            results=[SimpleNamespace(text=row['text'],score=row['confidence'],box=list(roi))]
                            white_cost_observation=dict(source='selected_card_face_enclosed_white_text',
                                text=row['text'],confidence=row['confidence'],roi=list(roi),
                                raw_results=raw,boundary_components=boundary,enclosed_components=enclosed)
    bundle=getattr(context,'gkh_hif_opening_state',None)
    if bundle is not None and 0<=candidate.slot<len(bundle.state.deck.hand):
        from live_decision.high_score.kernel import CardCostKind
        definition=bundle.kernel.cards[bundle.state.deck.hand[candidate.slot].card_id]
        if definition.cost_kind in (CardCostKind.CONCENTRATION, CardCostKind.GOOD_CONDITION):
            numeric={abs(int(row.text.strip().replace('−','-'))) for row in (*card_face_readings,*results)
                if _valid_ocr_result(row) and re.fullmatch(r'[-−]?\d{1,3}',row.text.strip())}
            if len(numeric)>1:
                return dict(accepted=False,value=None,confidence=0.,
                    reason='selected_effective_cost_concentration_contradiction',
                    roi=list(roi),candidate_values=sorted(numeric))
    semantic_results: list[tuple[Any, str, int, list[int]]] = []
    for result in results:
        if not _valid_ocr_result(result):
            continue
        raw_box = list(getattr(result, "box"))
        center_x = raw_box[0] + raw_box[2] / 2.0
        center_y = raw_box[1] + raw_box[3] / 2.0
        text = str(getattr(result, "text", "")).strip().replace("−", "-")
        if not (
            roi_x <= center_x < roi_x + roi_width
            and roi_y <= center_y < roi_y + roi_height
            and re.fullmatch(r"^(?:0|-\d{1,3})$", text) is not None
        ):
            continue
        value = abs(int(text))
        if 0 <= value <= 100:
            semantic_results.append((result, text, value, raw_box))
    distinct_values = {item[2] for item in semantic_results}
    if not semantic_results and bundle is not None and 0<=candidate.slot<len(bundle.state.deck.hand):
        from live_decision.hif_skill_support import (
            _qualified,
            _panel_bounds,
            _read_detail_title,
            _read_printed_cost,
        )
        from live_decision.high_score.kernel import CardCostKind

        definition=bundle.kernel.cards[bundle.state.deck.hand[candidate.slot].card_id]
        title_roi=locate_hif_selected_detail_title(image)
        if (definition.cost_kind in (CardCostKind.CONCENTRATION,CardCostKind.GOOD_CONDITION)
                and title_roi is not None):
            panel=_panel_bounds(image,title_roi)
            title_rows,raw_title_rows,excluded=_read_detail_title(context,image,title_roi,panel)
            tx,ty,tw,th=title_roi
            fee_roi=(tx+tw-2,ty,56,th)
            fees,raw_fees,source=_read_printed_cost(context,image,fee_roi,require_signed=True)
            if (len(title_rows)==1 and _qualified(title_rows,title_roi)
                    and re.sub(r'\s+','',title_rows[0]['text'])==re.sub(r'\s+','',definition.name)
                    and len(fees)==1 and _qualified(fees,fee_roi)
                    and re.fullmatch(r'[-−][1-9]\d{0,2}',fees[0]['text'].strip())):
                value=abs(int(fees[0]['text'].strip().replace('−','-')))
                if all(abs(int(row.text.strip().replace('−','-')))==value
                       for row in card_face_readings if _valid_ocr_result(row)
                       and re.fullmatch(r'[-−]?\d{1,3}',row.text.strip())):
                    return dict(accepted=True,value=value,
                        confidence=min(title_rows[0]['confidence'],fees[0]['confidence']),
                        reason='selected_effective_cost_named_header',text=fees[0]['text'],
                        box=fees[0]['box'],roi=list(fee_roi),observed_result_count=1,
                        corroboration=dict(source='selected_named_typed_cost_header',
                            title_rows=title_rows,raw_title_rows=raw_title_rows,
                            excluded_title_pictogram_boxes=excluded,fee_rows=fees,raw_fee_rows=raw_fees,
                            fee_source=source,card_face_roi=roi))
    if len(distinct_values) != 1:
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": (
                "selected_effective_cost_missing"
                if not semantic_results
                else "selected_effective_cost_ambiguous"
            ),
            "candidate_values": sorted(distinct_values),
            "observed_result_count": len(results),
            "roi": roi,
        }
    best, text, value, box = max(
        semantic_results,
        key=lambda item: float(getattr(item[0], "score", 0.0)),
    )
    corroboration = white_cost_observation
    bundle=getattr(context,'gkh_hif_opening_state',None)
    if float(best.score)<.95 and bundle is not None and text=='0':
        node='HifSelectedZeroCost'
        area=[roi[0]+6,roi[1]+2,30,34]
        if zero_probe_rows is None:
            detail=context.run_recognition(node,image,pipeline_override={node:dict(
                recognition='OCR',roi=area,only_rec=True,expected='.+')})
            rows=list(getattr(detail,'filtered_results',()) or ()) if detail and detail.hit else []
        else:
            rows=zero_probe_rows
        if (len(rows)==1 and _valid_ocr_result(rows[0]) and rows[0].text.strip()=='0'
                and .95<=rows[0].score<=1):
            corroboration=dict(source='selected_zero_cost_card_face',
                card_face_confidence=float(best.score),roi=area,text=rows[0].text)
            best=rows[0]
    if (float(best.score)<.95 and bundle is not None
            and 0<=candidate.slot<len(bundle.state.deck.hand)):
        from live_decision.high_score.kernel import CardCostKind
        definition=bundle.kernel.cards[bundle.state.deck.hand[candidate.slot].card_id]
        if definition.cost_kind in (CardCostKind.CONCENTRATION, CardCostKind.GOOD_CONDITION):
            # The selected information bubble is already visible. Read its
            # full title and signed fee independently; never substitute the
            # catalog's numeric cost for visible evidence.
            observed=[]
            for suffix,area in (('Title',[240,558,248,36]),('Fee',[577,558,35,34])):
                node='HifSelectedConcentration'+suffix
                detail=context.run_recognition(node,image,pipeline_override={node:dict(
                    recognition='OCR',roi=area,only_rec=True,expected='.+')})
                rows=list(getattr(detail,'filtered_results',()) or ()) if detail and detail.hit else []
                observed.append(rows[0] if len(rows)==1 and _valid_ocr_result(rows[0]) else None)
            title,fee=observed
            if (title is not None and fee is not None and min(title.score,fee.score)>=.95
                    and re.sub(r'\s+','',title.text)==re.sub(r'\s+','',definition.name)
                    and fee.text.strip().replace('−','-')==text):
                corroboration=dict(source='selected_named_information_bubble',
                    card_face_confidence=float(best.score),title=title.text,
                    title_confidence=float(title.score),fee=fee.text,fee_confidence=float(fee.score))
                best=SimpleNamespace(score=min(title.score,fee.score))
    return {
        "accepted": True,
        "value": value,
        "confidence": float(getattr(best, "score")),
        "reason": "selected_effective_cost_card_face",
        "text": text,
        "box": box,
        "roi": roi,
        "observed_result_count": len(results),
        **({'corroboration':corroboration} if corroboration else {}),
    }


def _build_red_only_resource_delta_image(
    image: Any,
    roi: list[int],
    *,
    separate_background_specks: bool = False,
    allow_subcomponent_zero: bool = False,
) -> tuple[Any | None, dict[str, Any]]:
    """Isolate one HUD red delta while preserving clean-zero evidence."""

    evidence: dict[str, Any] = {
        "accepted": False,
        "reason": "resource_delta_red_evidence_unreadable",
        "raw_red_pixel_count": None,
        "red_present": None,
        "pixel_count": 0,
        "box": None,
        "component_count": 0,
        "bottom_guard_red_pixels": None,
        "components": [],
        "roi": list(roi) if isinstance(roi, list) else None,
    }
    try:
        import cv2
        import numpy as np

        source = np.asarray(image)
        if (
            not isinstance(roi, list)
            or len(roi) != 4
            or any(type(value) is not int for value in roi)
        ):
            raise ValueError("invalid resource-delta ROI")
        x, y, width, height = roi
        if (
            width <= 0
            or height <= 0
            or source.ndim != 3
            or source.shape[2] < 3
            or source.shape[0] < y + height + 1
            or source.shape[1] < x + width
        ):
            raise ValueError("frame is smaller than the resource-delta region")
        region = source[y : y + height, x : x + width]
        bottom_guard = source[y + height : y + height + 1, x : x + width]

        def red_mask(pixels: Any) -> Any:
            blue = pixels[..., 0].astype(np.int16)
            green = pixels[..., 1].astype(np.int16)
            red = pixels[..., 2].astype(np.int16)
            return (
                (red >= 240)
                & ((red - green) >= 32)
                & ((red - blue) >= 32)
                & (np.abs(green - blue) <= 12)
            )

        raw_mask = red_mask(region)
        raw_red_pixel_count = int(np.count_nonzero(raw_mask))
        bottom_guard_red_pixels = int(np.count_nonzero(red_mask(bottom_guard)))
        evidence.update(
            {
                "raw_red_pixel_count": raw_red_pixel_count,
                "red_present": raw_red_pixel_count > 0,
                "bottom_guard_red_pixels": bottom_guard_red_pixels,
            }
        )
        if raw_red_pixel_count == 0:
            evidence["reason"] = "resource_delta_red_evidence_absent"
            return None, evidence
        if bottom_guard_red_pixels:
            raise ValueError("red resource-delta evidence crosses the guard row")
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            raw_mask.astype(np.uint8),
            connectivity=8,
        )
        # Preserve every raw pixel. Fewer pixels than even one accepted
        # component cannot be a signed number; consensus still requires two
        # entirely clear frames and independently unchanged HUD values.
        if allow_subcomponent_zero and raw_red_pixel_count < LIVE_RESOURCE_DELTA_RED_COMPONENT_MIN_AREA:
            points = np.argwhere(raw_mask)
            if all(0 < px < width-1 and 0 < py < height-1 for py, px in points):
                evidence['subcomponent_specks'] = [
                    [x+int(px), y+int(py)] for py, px in points]
        components: list[dict[str, Any]] = []
        retained = np.zeros_like(raw_mask)
        for component_index in range(1, component_count):
            local_x, local_y, local_width, local_height, area = (
                int(value) for value in stats[component_index]
            )
            if area < LIVE_RESOURCE_DELTA_RED_COMPONENT_MIN_AREA:
                continue
            retained[labels == component_index] = True
            components.append(
                {
                    "box": [local_x, local_y, local_width, local_height],
                    "area": area,
                }
            )
        if separate_background_specks:
            specks = [row for row in components
                if row['area'] <= 3 and max(row['box'][2:]) <= 2]
            word = sorted((row for row in components if row not in specks),
                key=lambda row:row['box'][0])
            # A signed nonzero word must already have complete digits. Tiny
            # disconnected pixels wholly above/below its line are background,
            # not a new numeral. No word means no exclusion and no clean zero.
            if specks and 2 <= len(word) <= 4 and all(row['box'][3] >= 10 for row in word[1:]):
                top = min(row['box'][1] for row in word)
                bottom = max(row['box'][1]+row['box'][3] for row in word)
                if all(row['box'][1]+row['box'][3] <= top-2
                       or row['box'][1] >= bottom+2 for row in specks):
                    for row in specks:
                        sx,sy,sw,sh = row['box']
                        retained[sy:sy+sh,sx:sx+sw] = False
                    evidence['excluded_background_components'] = [dict(
                        box=[x+row['box'][0],y+row['box'][1],*row['box'][2:]],
                        area=row['area']) for row in specks]
                    components = word
        red_y, red_x = np.where(retained)
        if red_x.size == 0:
            raise ValueError("red resource-delta components were rejected")
        pixel_count = int(red_x.size)
        local_box = [
            int(red_x.min()),
            int(red_y.min()),
            int(red_x.max() - red_x.min() + 1),
            int(red_y.max() - red_y.min() + 1),
        ]
        if (
            pixel_count < LIVE_RESOURCE_DELTA_RED_MIN_PIXELS
            or pixel_count > LIVE_RESOURCE_DELTA_RED_MAX_PIXELS
            or local_box[2] < LIVE_RESOURCE_DELTA_RED_MIN_WIDTH
            or local_box[3] < LIVE_RESOURCE_DELTA_RED_MIN_HEIGHT
        ):
            raise ValueError("red resource-delta glyph evidence is too small")
        components.sort(key=lambda component: component["box"][0])
        if not 2 <= len(components) <= 4:
            raise ValueError("red resource-delta component count is invalid")
        if any(
            component["box"][0] == 0
            or component["box"][1] == 0
            or component["box"][0] + component["box"][2] == width
            for component in components
        ):
            raise ValueError("red resource-delta component touches the boundary")
        minus = components[0]
        minus_x, _minus_y, minus_width, minus_height = minus["box"]
        if (
            not 6 <= minus_width <= 18
            or not 2 <= minus_height <= 7
            or minus_width / minus_height < 2.0
            or not 12 <= minus["area"] <= 80
        ):
            raise ValueError("red resource-delta minus component is invalid")
        digits = components[1:]
        if (
            any(component["box"][0] < minus_x + minus_width for component in digits)
            or max(component["box"][1] + component["box"][3] for component in digits)
            - min(component["box"][1] for component in digits)
            < 10
        ):
            raise ValueError("red resource-delta digit components are invalid")
        red_only = np.full_like(source, 255)
        isolated = np.full_like(region, 255)
        isolated[retained] = 0
        red_only[y : y + height, x : x + width] = isolated
        evidence.update(
            {
                "accepted": True,
                "reason": "resource_delta_red_pixels_isolated",
                "pixel_count": pixel_count,
                "box": [
                    x + local_box[0],
                    y + local_box[1],
                    local_box[2],
                    local_box[3],
                ],
                "component_count": len(components),
                "components": [
                    {
                        "box": [
                            x + component["box"][0],
                            y + component["box"][1],
                            component["box"][2],
                            component["box"][3],
                        ],
                        "area": component["area"],
                    }
                    for component in components
                ],
            }
        )
        return red_only, evidence
    except (AttributeError, ImportError, TypeError, ValueError):
        if evidence.get("red_present") is True:
            evidence["reason"] = "resource_delta_red_evidence_invalid"
        return None, evidence


def _prewarm_live_resource_evidence_dependencies() -> None:
    """Load the exact native resource-evidence path before an authoritative frame."""

    import cv2
    import numpy as np

    empty_mask = np.zeros((1, 1), dtype=np.uint8)
    cv2.connectedComponentsWithStats(empty_mask, connectivity=8)


def _live_resource_delta_roi(context, name, roi):
    # HIF places the stamina delta ten pixels higher than the NIA HUD.
    # Keep the full glyph inside the original red-component boundary guard.
    if (name == "stamina" and roi == LIVE_STAMINA_PREVIEW_DELTA_ROI
            and isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext)):
        return [568, 240, 64, 45]
    if (name == 'shield' and roi == LIVE_SHIELD_PREVIEW_DELTA_ROI
            and isinstance(getattr(context, 'gkh_hif_live_context', None), HifLiveRunContext)):
        # The floating red delta reaches y=235 in 1356/1357. Use the same
        # complete bubble as blue shield deltas; keep its real guard row.
        return [648, 200, 72, 44]
    return roi


def _ocr_resource_preview_delta(
    context: Context,
    image: Any,
    name: str,
    roi: list[int],
) -> dict[str, Any]:
    """Read one signed HUD delta; absence remains distinct from invalid red."""

    roi = _live_resource_delta_roi(context, name, roi)
    red_only_image, red_evidence, roi = _live_signed_delta_image(context,image,name,roi)
    if red_only_image is None:
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": red_evidence["reason"],
            "red_evidence": red_evidence,
            "roi": list(roi),
        }
    node_name = f"Task250Preview{name.title()}Delta"
    detail = context.run_recognition(
        node_name,
        red_only_image,
        pipeline_override={
            node_name: {
                "recognition": "OCR",
                "roi": roi,
                "expected": r"^\+\d{1,3}$" if red_evidence.get('color')=='blue' else r"^[\-−]\d{1,3}$",
            }
        },
    )
    results = (
        list(getattr(detail, "filtered_results", ()) or ())
        if detail and detail.hit
        else []
    )
    return _parse_resource_delta_ocr_results(
        results,
        red_evidence=red_evidence,
        roi=roi,
    )


def _parse_resource_delta_ocr_results(
    results: list[Any],
    *,
    red_evidence: dict[str, Any],
    roi: list[int],
) -> dict[str, Any]:
    """Apply signed-delta semantics after either direct or mosaic OCR."""

    roi_x, roi_y, roi_width, roi_height = roi
    semantic_results: list[tuple[Any, str, int, list[int]]] = []
    for result in results:
        if not _valid_ocr_result(result):
            continue
        raw_box = list(getattr(result, "box"))
        center_x = raw_box[0] + raw_box[2] / 2.0
        center_y = raw_box[1] + raw_box[3] / 2.0
        components = red_evidence.get("components")
        all_components_bound = bool(
            isinstance(components, list)
            and components
            and all(
                raw_box[0]
                <= component["box"][0] + component["box"][2] / 2.0
                < raw_box[0] + raw_box[2]
                and raw_box[1]
                <= component["box"][1] + component["box"][3] / 2.0
                < raw_box[1] + raw_box[3]
                for component in components
            )
        )
        text = str(getattr(result, "text", "")).strip().replace("−", "-")
        if not (
            roi_x <= center_x < roi_x + roi_width
            and roi_y <= center_y < roi_y + roi_height
            and all_components_bound
            and re.fullmatch(r"^\+\d{1,3}$" if red_evidence.get('color')=='blue' else r"^-\d{1,3}$", text) is not None
            and red_evidence.get("component_count") == len(text)
        ):
            continue
        value = int(text)
        if (1 <= value <= 100) if red_evidence.get('color')=='blue' else (-100 <= value <= -1):
            semantic_results.append((result, text, value, raw_box))
    distinct_values = {item[2] for item in semantic_results}
    if len(distinct_values) != 1:
        return {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": (
                "resource_delta_ocr_missing"
                if not semantic_results
                else "resource_delta_ocr_ambiguous"
            ),
            "candidate_values": sorted(distinct_values),
            "observed_result_count": len(results),
            "red_evidence": red_evidence,
            "roi": list(roi),
        }
    best, text, value, box = max(
        semantic_results,
        key=lambda item: float(getattr(item[0], "score", 0.0)),
    )
    return {
        "accepted": True,
        "value": value,
        "confidence": float(getattr(best, "score")),
        "reason": "resource_delta_signed_ocr",
        "text": text,
        "box": box,
        "observed_result_count": len(results),
        "red_evidence": red_evidence,
        "roi": list(roi),
    }


def _live_signed_delta_image(context,image,name,roi):
    if name in ('stamina','shield') and isinstance(getattr(context,'gkh_hif_live_context',None),HifLiveRunContext):
        from live_decision.hif_visual import hif_blue_resource_delta
        # Positive stamina uses the same white bubble as genki, centered below
        # the heart. Its complete plus sign lies left of the red-cost ROI.
        blue_roi=[544,240,88,45] if name=='stamina' else [648,200,72,44]
        isolated,evidence=hif_blue_resource_delta(image,blue_roi)
        if isolated is not None:
            return isolated,evidence,blue_roi
    small_hif = (isinstance(getattr(context,'gkh_hif_live_context',None),HifLiveRunContext)
        and getattr(context,'gkh_hif_client_size',None) in ((540,960),(541,961),(541,962),(542,963)))
    isolated,evidence=_build_red_only_resource_delta_image(image,roi,
        separate_background_specks=small_hif,
        allow_subcomponent_zero=isinstance(getattr(context,'gkh_hif_live_context',None),HifLiveRunContext))
    return isolated,evidence,roi


def _live_delta_mosaic_sources(context, image):
    delta_observations: dict[str, dict[str, Any]] = {}
    delta_sources: list[dict[str, Any]] = []
    delta_masks: dict[str, dict[str, Any]] = {}
    for column, (name, roi) in enumerate(
        (
            ("stamina", LIVE_STAMINA_PREVIEW_DELTA_ROI),
            ("shield", LIVE_SHIELD_PREVIEW_DELTA_ROI),
        )
    ):
        roi = _live_resource_delta_roi(context, name, roi)
        red_only, red_evidence, roi = _live_signed_delta_image(context,image,name,roi)
        delta_masks[name] = red_evidence
        if red_only is None:
            delta_observations[name] = {
                "accepted": False,
                "value": None,
                "confidence": 0.0,
                "reason": red_evidence["reason"],
                "red_evidence": red_evidence,
                "roi": list(roi),
                "acquisition_mode": "explicit_red_mask",
            }
        else:
            delta_sources.append(
                {
                    "key": f"{name}_delta",
                    "row": 0,
                    "column": column,
                    "roi": roi,
                    "image": red_only,
                    "expected": r"^\+\d{1,3}$" if red_evidence.get('color')=='blue' else r"^[\-−]\d{1,3}$",
                }
            )
    return delta_observations, delta_sources, delta_masks


def _observe_authoritative_live_ocr_bundle(
    context: Context,
    image: Any,
    candidates: tuple[Any, ...],
    *,
    frame_id: str,
    absolute_min_confidence: float,
    preview_delta_min_confidence: float,
    require_fields: bool,
) -> dict[str, Any]:
    """Acquire one geometry-bound common mosaic plus cached resource evidence."""

    started_at = time.perf_counter_ns()
    hif_label_reader = isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext)
    try:
        candidate_signature = _live_candidate_signature(candidates)
        slots = tuple(candidate.slot for candidate in candidates)
        if slots != (0, 1, 2) and not (hif_label_reader and slots in ((0,), (0, 1), (0, 1, 2, 3), (0, 1, 2, 3, 4))):
            raise ValueError("authoritative_live_mosaic_requires_three_slots")
        shape = tuple(getattr(image, "shape", ()))
        select_rois = (
            [list(roi) for roi in hif_select_label_rois(shape, candidates)]
            if hif_label_reader
            else [list(live_select_label_box(shape, candidate)) for candidate in candidates]
        )
    except (AttributeError, LiveSelectionError, TypeError, ValueError):
        return {
            "supported": False,
            "accepted": False,
            "reason": "authoritative_live_mosaic_source_unavailable",
            "frame_id": frame_id,
            "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
        }

    action_order = _locate_live_self_rank(context, image, candidates=candidates)
    score_roi = action_order.get("score_roi") if action_order.get("accepted") else None
    sources = [
        {
            "key": f"select_{slot}",
            "row": 0,
            "column": slot,
            "roi": select_rois[slot],
            "image": image,
            "expected": r"^SELECT$",
        }
        for slot in (() if hif_label_reader else range(3))
    ]
    # HIF has animated stage pixels below unselected cards. The union digit
    # detector can read those as numbers inside a SELECT tile. Reuse the Live hand recognizer's
    # exact full-glyph reader here, independently from the numeric mosaic.
    hif_select_slots = _locate_hif_selected_slots(
        context, image, candidates, reuse_observation=True) if hif_label_reader else None
    resampled_shield = hif_label_reader and tuple(getattr(context, 'gkh_hif_client_size', ())) in {
        (540, 960), (541, 961), (541, 962), (542, 962)}
    direct_hif_shield = (hif_label_reader and calibrated_live_client_size(
        getattr(context, 'gkh_hif_client_size', ())) == (480, 853))
    shield_roi = _live_field_roi(context, "shield", LIVE_SHIELD_ROI, image)
    shield_badge = _hif_shield_numeric_badge_roi(
        image, shield_roi, **({'allow_filled_bar': True} if resampled_shield else {})
    ) if hif_label_reader else None
    shield_image = image
    if resampled_shield:
        import cv2
        x, y, width, height = shield_roi
        shield_image = image.copy()
        shield_image[y:y+height, x:x+width] = cv2.cvtColor(
            image[y:y+height, x:x+width].min(axis=2), cv2.COLOR_GRAY2BGR)
        if shield_badge is not None and shield_badge.get('bar_cap_color') == 'gray':
            # Keep the complete original ROI and the independent badge in
            # separate mosaic cells. Only its proven non-numeric bar region
            # becomes the same white background as the mosaic padding.
            bx, by, bw, bh = shield_badge['excluded_bar_region']
            shield_image[by:by+bh, bx:bx+bw] = 255
    if shield_badge is not None and not direct_hif_shield:
        # HIF SELECT uses its independent glyph reader, leaving this fixed
        # cell free. Keep the complete shield crop in the same OCR batch.
        sources.append({
            "key": "shield_numeric_badge", "row": 0, "column": 0,
            "roi": shield_badge["roi"], "image": shield_image, "expected": r"^\d{1,3}$",
        })
    sources.extend(
        [
            {
                "key": "stamina",
                "row": 1,
                "column": 0,
                "roi": _live_field_roi(context, "stamina", LIVE_LAYOUT_V1["stamina"][0]),
                "image": image,
                "expected": LIVE_LAYOUT_V1["stamina"][1],
            },
            {
                "key": "shield",
                "row": 1,
                "column": 1,
                "roi": _live_field_roi(context, "shield", LIVE_SHIELD_ROI, image),
                "image": shield_image,
                "expected": r"^\d{1,3}$",
            },
            {
                "key": "turns_remaining",
                "row": 2,
                "column": 2,
                "roi": _live_field_roi(context, "turns_remaining", LIVE_LAYOUT_V1["turns_remaining"][0]),
                "image": image,
                "expected": LIVE_LAYOUT_V1["turns_remaining"][1],
            },
            {
                "key": "score_multiplier",
                "row": 3,
                "column": 0,
                "roi": _live_field_roi(context, "score_multiplier", LIVE_LAYOUT_V1["score_multiplier"][0]),
                "image": image,
                "expected": LIVE_LAYOUT_V1["score_multiplier"][1],
            },
        ]
    )
    if isinstance(score_roi, (list, tuple)) and len(score_roi) == 4:
        sources.append(
            {
                "key": "score",
                "row": 3,
                "column": 1,
                "roi": list(score_roi),
                "image": image,
                "expected": r"^\d+$",
            }
        )
    hif_deltas = None
    if hif_label_reader:
        hif_deltas = _live_delta_mosaic_sources(context, image)
        sources.extend({**source, "row": 2} for source in hif_deltas[1])
    if direct_hif_shield:
        # Native 480px HUD digits lose repeated strokes in the composed mosaic.
        # Choose the existing complete source ROI before recognition, never
        # select a favourable number after comparing competing observations.
        sources = [source for source in sources if source['key'] != 'shield']
    mosaic, placements, source_failure = _build_fixed_live_ocr_mosaic(sources)
    if mosaic is None:
        return {
            "supported": False,
            "accepted": False,
            "reason": source_failure,
            "frame_id": frame_id,
            "candidate_signature": candidate_signature,
            "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
        }
    common = _recognize_fixed_live_ocr_mosaic(
        context,
        mosaic,
        placements,
        node_name="Task250AuthoritativeCommonOcrMosaic",
    )
    if common.get("supported") is False:
        return {
            "supported": False,
            "accepted": False,
            "reason": "authoritative_live_mosaic_api_unavailable",
            "frame_id": frame_id,
            "candidate_signature": candidate_signature,
            "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
        }
    if common.get("accepted") is not True:
        return {
            "supported": True,
            "accepted": False,
            "reason": "authoritative_live_common_mosaic_contradiction",
            "frame_id": frame_id,
            "candidate_signature": candidate_signature,
            "common": common,
            "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
        }
    grouped = common["results"]
    if any(len(grouped.get(f"select_{slot}", [])) > 1 for slot in range(3)):
        return {
            "supported": True,
            "accepted": False,
            "reason": "authoritative_live_common_mosaic_contradiction",
            "frame_id": frame_id,
            "candidate_signature": candidate_signature,
            "common": common,
            "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
        }
    select_slots = hif_select_slots if hif_label_reader else tuple(
        slot for slot in range(3) if grouped.get(f"select_{slot}")
    )
    fields: dict[str, dict[str, Any]] = {}
    fallbacks: list[dict[str, Any]] = []
    for name, expected, roi in (
        ("stamina", LIVE_LAYOUT_V1["stamina"][1], LIVE_LAYOUT_V1["stamina"][0]),
        ("shield", r"^\d{1,3}$", LIVE_SHIELD_ROI),
    ):
        roi = _live_field_roi(context, name, roi, image)
        if name == 'shield' and direct_hif_shield:
            direct = _ocr_field(context,image,name,LIVE_SHIELD_ROI,r'^\d{1,3}$')
            fields[name] = dict(direct,acquisition_mode='direct_calibrated_hif_shield',
                source_acquisition_mode=direct.get('acquisition_mode'), roi=list(shield_roi))
            continue
        mosaic_field = _parse_live_field_ocr_results(
            grouped.get(name, []),
            name=name,
            expected=expected,
        )
        if (
            mosaic_field.get("reason") == "live_field_ocr_ambiguous"
            or len(grouped.get(name, [])) > 1
        ):
            return {
                "supported": True,
                "accepted": False,
                "reason": "authoritative_live_common_mosaic_contradiction",
                "frame_id": frame_id,
                "candidate_signature": candidate_signature,
                "common": common,
                "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
            }
        if name == "shield" and shield_badge is not None:
            badge_rows = grouped.get("shield_numeric_badge", [])
            numeric = _parse_live_field_ocr_results(
                badge_rows, name=name, expected=expected,
            )
            if (len(badge_rows) > 1
                    or numeric.get("reason") == "live_field_ocr_ambiguous"
                    or (numeric.get("accepted") is True
                        and mosaic_field.get("accepted") is True
                        and numeric.get("value") != mosaic_field.get("value"))):
                return {
                    "supported": True, "accepted": False,
                    "reason": "authoritative_live_common_mosaic_contradiction",
                    "frame_id": frame_id, "candidate_signature": candidate_signature,
                    "common": common, "fallback_conflict": {
                        "field": name, "mosaic_value": mosaic_field.get("value"),
                        "numeric_badge_value": numeric.get("value"),
                    },
                    "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
                }
            badge_confidence = numeric.get("confidence")
            if (numeric.get("accepted") is True
                    and type(badge_confidence) in (int, float)
                    and math.isfinite(float(badge_confidence))
                    and max(.95, absolute_min_confidence) <= badge_confidence <= 1):
                fields[name] = dict(
                    numeric, acquisition_mode="common_mosaic_shield_numeric_badge",
                    roi=shield_badge["roi"], badge_boundary=shield_badge,
                    original_observation=mosaic_field,
                )
                continue
        confidence = mosaic_field.get("confidence")
        trusted = bool(
            mosaic_field.get("accepted") is True
            and not isinstance(confidence, bool)
            and isinstance(confidence, (int, float))
            and math.isfinite(float(confidence))
            and float(confidence) >= absolute_min_confidence
        )
        if trusted:
            mosaic_field["acquisition_mode"] = "common_mosaic"
            fields[name] = mosaic_field
        else:
            fallback_reason = (
                "low_confidence"
                if mosaic_field.get("accepted") is True
                else "missing"
            )
            direct = _ocr_field(context, image, name, list(roi), expected)
            if (
                fallback_reason == "low_confidence"
                and direct.get("accepted") is True
                and direct.get("value") != mosaic_field.get("value")
            ):
                return {
                    "supported": True,
                    "accepted": False,
                    "reason": "authoritative_live_common_mosaic_contradiction",
                    "frame_id": frame_id,
                    "candidate_signature": candidate_signature,
                    "common": common,
                    "fallback_conflict": {
                        "field": name,
                        "mosaic_value": mosaic_field.get("value"),
                        "direct_value": direct.get("value"),
                        "mosaic_observation": dict(mosaic_field),
                        "direct_observation": dict(direct),
                    },
                    "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
                }
            direct["acquisition_mode"] = "direct_missing_fallback"
            fields[name] = direct
            fallbacks.append({"field": name, "reason": fallback_reason})

    for name in ("turns_remaining", "score_multiplier"):
        expected = LIVE_LAYOUT_V1[name][1]
        parsed = _parse_live_field_ocr_results(
            grouped.get(name, []),
            name=name,
            expected=expected,
        )
        if (
            parsed.get("reason") == "live_field_ocr_ambiguous"
            or len(grouped.get(name, [])) > 1
        ):
            return {
                "supported": True,
                "accepted": False,
                "reason": "authoritative_live_common_mosaic_contradiction",
                "frame_id": frame_id,
                "candidate_signature": candidate_signature,
                "common": common,
                "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
            }
        if parsed.get("accepted") is not True and require_fields:
            parsed = _ocr_field(
                context,
                image,
                name,
                LIVE_LAYOUT_V1[name][0],
                expected,
            )
            parsed["acquisition_mode"] = "direct_missing_fallback"
            fallbacks.append({"field": name, "reason": "missing"})
        else:
            parsed["acquisition_mode"] = "common_mosaic"
        fields[name] = parsed

    score = _parse_live_field_ocr_results(
        grouped.get("score", []),
        name="score",
        expected=r"^\d+$",
    )
    if (
        score.get("reason") == "live_field_ocr_ambiguous"
        or len(grouped.get("score", [])) > 1
    ):
        return {
            "supported": True,
            "accepted": False,
            "reason": "authoritative_live_common_mosaic_contradiction",
            "frame_id": frame_id,
            "candidate_signature": candidate_signature,
            "common": common,
            "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
        }
    if score.get("accepted") is True:
        score["acquisition_mode"] = "common_mosaic_wide"
    if score.get("accepted") is not True and require_fields:
        score = _ocr_live_score(context, image, action_order,
            **({'observed_wide_miss': score} if hif_label_reader else {}))
        score["acquisition_mode"] = "direct_missing_fallback"
        fallbacks.append({"field": "score", "reason": "missing"})
    fields["score"] = score

    delta_observations, delta_sources, delta_masks = (
        hif_deltas if hif_label_reader else _live_delta_mosaic_sources(context, image))
    delta_common: dict[str, Any] | None = None
    if delta_sources:
        if hif_label_reader:
            delta_common = common
        else:
            delta_mosaic, delta_placements, delta_source_failure = (
                _build_fixed_live_ocr_mosaic(
                    delta_sources,
                    cell_size=LIVE_AUTHORITATIVE_DELTA_MOSAIC_CELL,
                    mosaic_shape=LIVE_AUTHORITATIVE_DELTA_MOSAIC_SHAPE,
                )
            )
            if delta_mosaic is None:
                return {
                    "supported": True,
                    "accepted": False,
                    "reason": delta_source_failure,
                    "frame_id": frame_id,
                    "candidate_signature": candidate_signature,
                    "common": common,
                    "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
                }
            delta_common = _recognize_fixed_live_ocr_mosaic(
                context,
                delta_mosaic,
                delta_placements,
                node_name="Task250AuthoritativeDeltaOcrMosaic",
                cell_size=LIVE_AUTHORITATIVE_DELTA_MOSAIC_CELL,
                mosaic_shape=LIVE_AUTHORITATIVE_DELTA_MOSAIC_SHAPE,
            )
        if delta_common.get("supported") is False:
            return {
                "supported": False,
                "accepted": False,
                "reason": "authoritative_live_mosaic_api_unavailable",
                "frame_id": frame_id,
                "candidate_signature": candidate_signature,
                "common": common,
                "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
            }
        if delta_common.get("accepted") is not True:
            return {
                "supported": True,
                "accepted": False,
                "reason": "authoritative_live_delta_mosaic_contradiction",
                "frame_id": frame_id,
                "candidate_signature": candidate_signature,
                "common": common,
                "delta_common": delta_common,
                "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
            }
        for name, roi in (
            ("stamina", LIVE_STAMINA_PREVIEW_DELTA_ROI),
            ("shield", LIVE_SHIELD_PREVIEW_DELTA_ROI),
        ):
            roi = _live_resource_delta_roi(context, name, roi)
            if f"{name}_delta" not in delta_common["results"]:
                continue
            parsed = _parse_resource_delta_ocr_results(
                delta_common["results"][f"{name}_delta"],
                red_evidence=delta_masks[name],
                roi=delta_masks[name].get('roi',roi),
            )
            if (
                parsed.get("reason") == "resource_delta_ocr_ambiguous"
                or len(delta_common["results"][f"{name}_delta"]) > 1
            ):
                return {
                    "supported": True,
                    "accepted": False,
                    "reason": "authoritative_live_delta_mosaic_contradiction",
                    "frame_id": frame_id,
                    "candidate_signature": candidate_signature,
                    "common": common,
                    "delta_common": delta_common,
                    "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
                }
            confidence = parsed.get("confidence")
            mosaic_delta = dict(parsed)
            if (
                parsed.get("accepted") is not True
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or float(confidence) < preview_delta_min_confidence
            ):
                parsed = _ocr_resource_preview_delta(context, image, name, roi)
                if (
                    mosaic_delta.get("accepted") is True
                    and parsed.get("accepted") is True
                    and parsed.get("value") != mosaic_delta.get("value")
                ):
                    return {
                        "supported": True,
                        "accepted": False,
                        "reason": "authoritative_live_delta_mosaic_contradiction",
                        "frame_id": frame_id,
                        "candidate_signature": candidate_signature,
                        "common": common,
                        "delta_common": delta_common,
                        "fallback_conflict": {
                            "field": f"{name}_delta",
                            "mosaic_value": mosaic_delta.get("value"),
                            "direct_value": parsed.get("value"),
                        },
                        "elapsed_ms": (
                            time.perf_counter_ns() - started_at
                        )
                        / 1_000_000.0,
                    }
                parsed["acquisition_mode"] = "direct_missing_fallback"
                fallbacks.append({"field": f"{name}_delta", "reason": "missing"})
            else:
                parsed["acquisition_mode"] = "delta_mosaic"
            delta_observations[name] = parsed

    return {
        "supported": True,
        "accepted": True,
        "reason": "authoritative_live_ocr_bundle_acquired",
        "frame_id": frame_id,
        "candidate_signature": candidate_signature,
        "select_slots": select_slots,
        "select_acquisition_mode": "hif_exact_label" if hif_label_reader else "common_mosaic",
        "fields": fields,
        "action_order": action_order,
        "delta_observations": delta_observations,
        "common": common,
        "delta_common": delta_common,
        "fallbacks": fallbacks,
        "absolute_min_confidence": absolute_min_confidence,
        "elapsed_ms": (time.perf_counter_ns() - started_at) / 1_000_000.0,
    }


def _authoritative_live_ocr_bundle_bound(
    bundle: dict[str, Any],
    *,
    frame_id: str,
    candidates: tuple[Any, ...],
) -> bool:
    try:
        return bool(
            bundle.get("accepted") is True
            and bundle.get("frame_id") == frame_id
            and tuple(bundle.get("candidate_signature", ()))
            == _live_candidate_signature(candidates)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _observe_live_fields(
    context: Context,
    image: Any,
    *,
    commit_turn: bool = True,
    authoritative_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read Live fields, optionally isolating the turn-tracker observation."""

    cached_fields = (
        authoritative_evidence.get("fields")
        if isinstance(authoritative_evidence, dict)
        and authoritative_evidence.get("accepted") is True
        else None
    )
    if isinstance(cached_fields, dict):
        fields = {
            name: dict(cached_fields.get(name, {})) for name in LIVE_LAYOUT_V1
        }
        fields["shield"] = dict(cached_fields.get("shield", {}))
    else:
        fields = {
            name: _ocr_field(context, image, name, roi, expected)
            for name, (roi, expected) in LIVE_LAYOUT_V1.items()
        }
        fields["shield"] = _ocr_field(
            context,
            image,
            "shield",
            LIVE_SHIELD_ROI,
            r"^\d{1,3}$",
        )
    hif_run = getattr(context, "gkh_hif_live_context", None)
    round_context = getattr(context, "gkh_hif_round_context", None)
    if isinstance(hif_run, HifLiveRunContext) and round_context is not None:
        from live_decision.layout import infer_dmm_turn
        from live_decision.hif_session import resolve_hif_round_context

        _, total_turns = resolve_hif_round_context(round_context,
            run_id=hif_run.run_id, revision=hif_run.revision)
        fields["turn"] = (infer_dmm_turn(fields["turns_remaining"]["value"], total_turns)
            if fields["turns_remaining"]["accepted"]
            else {"accepted": False, "value": None, "confidence": 0.0})
    else:
        turn_tracker = TURN_TRACKER if commit_turn else replace(TURN_TRACKER)
        fields["turn"] = (
            turn_tracker.observe(fields["turns_remaining"].get("value"), time.monotonic())
            if fields["turns_remaining"]["accepted"]
            else {"accepted": False, "value": None, "confidence": 0.0}
        )
    # The heart value is current stamina.  The horizontal bar has a green
    # stamina base with a blue shield overlay, while the upper-right number is
    # the shield value.  The legacy geometry helper currently merges both
    # saturated layers and has not been calibrated for partial stamina with a
    # positive shield, so it must not yet synthesize max_stamina.
    fields["max_stamina"] = {
        "accepted": False,
        "value": None,
        "confidence": 0.0,
        "reason": "max_stamina_bar_overlay_uncalibrated",
    }
    if isinstance(hif_run, HifLiveRunContext):
        stamina = fields["stamina"].get("value")
        if fields["stamina"].get("accepted") and type(stamina) is int and stamina <= hif_run.max_stamina:
            fields["max_stamina"] = {
                "accepted": True, "value": hif_run.max_stamina, "confidence": 1.0,
                "reason": "verified_hif_route_context", "run_id": hif_run.run_id,
                "revision": hif_run.revision,
            }
        elif fields["stamina"].get("accepted") and type(stamina) is int:
            fields["stamina"] = {
                "accepted": False, "value": None, "confidence": 0.0,
                "reason": "hif_run_stamina_contradiction",
                "observed_value": stamina, "run_max_stamina": hif_run.max_stamina,
            }
    fields["action_order"] = (
        dict(authoritative_evidence.get("action_order", {}))
        if isinstance(cached_fields, dict)
        else _locate_live_self_rank(context, image)
    )
    if isinstance(cached_fields, dict):
        fields["score"] = dict(cached_fields.get("score", {}))
    elif fields["action_order"]["accepted"]:
        fields["score"] = _ocr_live_score(context, image, fields["action_order"])
    else:
        fields["score"] = {"accepted": False, "value": None, "confidence": 0.0}
    for unresolved in ("items", "field_effects", "statuses"):
        fields[unresolved] = {"accepted": False, "value": None, "confidence": 0.0}
    return fields


def _observe_debug_preprobe_turns_remaining(
    context: Context,
    image: Any,
) -> dict[str, Any]:
    """Read probe metadata without mutating the authoritative turn tracker."""

    return _ocr_field(
        context,
        image,
        "turns_remaining",
        LIVE_LAYOUT_V1["turns_remaining"][0],
        LIVE_LAYOUT_V1["turns_remaining"][1],
    )


def _observe_live_resource_absolutes(
    context: Context,
    frames: tuple[Any, ...],
    *,
    authoritative_evidence: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Read absolute stamina and shield on the available battle observation."""

    if len(frames) != 3 and not (len(frames) == 1 and getattr(context, 'gkh_hif_upstream_card_ui', False) is True):
        return {
            "accepted": False,
            "stamina": None,
            "shield": None,
            "confidence": 0.0,
            "reason": "live_resource_absolute_frames_incomplete",
            "observations": [],
        }
    if authoritative_evidence is not None and len(authoritative_evidence) == len(frames):
        observations = tuple(
            {
                "stamina": dict(evidence.get("fields", {}).get("stamina", {})),
                "shield": dict(evidence.get("fields", {}).get("shield", {})),
            }
            for evidence in authoritative_evidence
        )
    else:
        observations = tuple(
            {
                "stamina": _ocr_field(
                    context,
                    frame,
                    "stamina",
                    LIVE_LAYOUT_V1["stamina"][0],
                    LIVE_LAYOUT_V1["stamina"][1],
                ),
                "shield": _ocr_field(
                    context,
                    frame,
                    "shield",
                    LIVE_SHIELD_ROI,
                    r"^\d{1,3}$",
                ),
            }
            for frame in frames
        )
    stamina_values = tuple(
        observation["stamina"].get("value") for observation in observations
    )
    shield_values = tuple(
        observation["shield"].get("value") for observation in observations
    )
    accepted = bool(
        all(
            observation[name].get("accepted") is True
            and type(observation[name].get("value")) is int
            for observation in observations
            for name in ("stamina", "shield")
        )
        and len(set(stamina_values)) == 1 and len(set(shield_values)) == 1
    )
    return {
        "accepted": accepted,
        "stamina": stamina_values[0] if accepted else None,
        "shield": shield_values[0] if accepted else None,
        "confidence": (
            min(
                float(observation[name].get("confidence", 0.0))
                for observation in observations
                for name in ("stamina", "shield")
            )
            if accepted
            else 0.0
        ),
        "reason": (
            "live_resource_absolutes_unanimous"
            if accepted
            else "live_resource_absolutes_unresolved"
        ),
        "observations": [
            {
                "stamina": dict(observation["stamina"]),
                "shield": dict(observation["shield"]),
            }
            for observation in observations
        ],
    }


def _resource_absolute_confidence_retry_evidence(
    resource_transaction: dict[str, Any],
    resource_baseline: dict[str, Any],
    threshold: Any,
) -> dict[str, Any] | None:
    """Freeze the one observed low-confidence absolute OCR outlier for an A retry."""

    def strict_confidence(value: Any) -> float | None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            return None
        return float(value)

    threshold_value = strict_confidence(threshold)
    transaction_confidence = strict_confidence(
        resource_transaction.get("resource_absolute_confidence")
    )
    baseline_confidence = strict_confidence(resource_baseline.get("confidence"))
    if (
        threshold_value is None
        or transaction_confidence is None
        or transaction_confidence >= threshold_value
        or resource_baseline.get("accepted") is not True
        or type(resource_baseline.get("stamina")) is not int
        or type(resource_baseline.get("shield")) is not int
        or baseline_confidence is None
        or baseline_confidence < threshold_value
        or resource_transaction.get("accepted") is not True
        or resource_transaction.get("reason")
        != "unselected_resource_transaction_conserved"
        or resource_transaction.get("selected_slot") is not None
        or resource_transaction.get("displayed_effective_cost") is not None
        or type(resource_transaction.get("stamina_delta")) is not int
        or resource_transaction.get("stamina_delta") != 0
        or type(resource_transaction.get("shield_delta")) is not int
        or resource_transaction.get("shield_delta") != 0
    ):
        return None
    conservation = resource_transaction.get("conservation")
    if (
        not isinstance(conservation, dict)
        or not conservation
        or any(value is not True for value in conservation.values())
    ):
        return None
    for name in ("stamina", "shield"):
        baseline_value = resource_baseline[name]
        if any(
            type(resource_transaction.get(key)) is not int
            or resource_transaction.get(key) != baseline_value
            for key in (f"preview_{name}", f"normalized_{name}")
        ):
            return None

    transaction_observations = resource_transaction.get("observations")
    absolutes = (
        transaction_observations.get("absolutes")
        if isinstance(transaction_observations, dict)
        else None
    )
    observations = absolutes.get("observations") if isinstance(absolutes, dict) else None
    if (
        not isinstance(absolutes, dict)
        or absolutes.get("accepted") is not True
        or not isinstance(observations, (list, tuple))
        or len(observations) != 3
    ):
        return None

    field_rois = {
        "stamina": LIVE_LAYOUT_V1["stamina"][0],
        "shield": LIVE_SHIELD_ROI,
    }
    field_evidence: dict[str, dict[str, Any]] = {}
    low_confidence_observations: list[dict[str, Any]] = []
    all_confidences: list[float] = []
    for name in ("stamina", "shield"):
        values: list[int] = []
        texts: list[str] = []
        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        roi_x, roi_y, roi_width, roi_height = field_rois[name]
        for frame_index, observation in enumerate(observations):
            field = observation.get(name) if isinstance(observation, dict) else None
            if not isinstance(field, dict) or field.get("accepted") is not True:
                return None
            value = field.get("value")
            text = field.get("text")
            raw_box = field.get("box")
            confidence = strict_confidence(field.get("confidence"))
            if (
                type(value) is not int
                or type(text) is not str
                or re.fullmatch(r"^\d{1,3}$", text) is None
                or int(text) != value
                or not isinstance(raw_box, (list, tuple))
                or len(raw_box) != 4
                or any(type(component) is not int for component in raw_box)
                or raw_box[2] <= 0
                or raw_box[3] <= 0
                or raw_box[0] < roi_x
                or raw_box[1] < roi_y
                or raw_box[0] + raw_box[2] > roi_x + roi_width
                or raw_box[1] + raw_box[3] > roi_y + roi_height
                or confidence is None
            ):
                return None
            normalized_box = tuple(raw_box)
            values.append(value)
            texts.append(text)
            boxes.append(normalized_box)
            confidences.append(confidence)
            all_confidences.append(confidence)
            if confidence < threshold_value:
                low_confidence_observations.append(
                    {
                        "field": name,
                        "frame_index": frame_index,
                        "confidence": confidence,
                    }
                )
        if len(set(values)) != 1 or len(set(texts)) != 1 or len(set(boxes)) != 1:
            return None
        if type(absolutes.get(name)) is not int or absolutes.get(name) != values[0]:
            return None
        field_evidence[name] = {
            "value": values[0],
            "text": texts[0],
            "box": list(boxes[0]),
            "confidences": confidences,
        }

    absolute_confidence = strict_confidence(absolutes.get("confidence"))
    overall_transaction_confidence = strict_confidence(
        resource_transaction.get("confidence")
    )
    minimum_confidence = min(all_confidences)
    if (
        len(all_confidences) != 6
        or len(low_confidence_observations) != 1
        or absolute_confidence != minimum_confidence
        or transaction_confidence != minimum_confidence
        or overall_transaction_confidence != minimum_confidence
    ):
        return None
    return {
        "reason": "single_resource_absolute_confidence_outlier",
        "threshold": threshold_value,
        "baseline_confidence": baseline_confidence,
        "minimum_confidence": minimum_confidence,
        "low_confidence_observation": low_confidence_observations[0],
        "fields": field_evidence,
    }


def _observe_live_resource_baseline(
    context: Context,
    frames: tuple[Any, ...],
    *,
    first_frame_candidates: tuple[Any, ...] | None = None,
    expected_hand_count: int | None = None,
) -> dict[str, Any]:
    """Freeze an unselected resource baseline before any identity-probe input."""

    frame_ids = [frame_identifier(frame) for frame in frames]
    absolutes = _observe_live_resource_absolutes(context, frames)
    fragment_reconciliations: list[dict[str, Any]] = []
    try:
        consensus, frame_candidates, select_evidence, geometry_stable = (
            _observe_live_selected_slot_consensus(
                context,
                frames,
                expected_selected_slot=None,
                first_frame_candidates=first_frame_candidates,
                reconciliation_events=fragment_reconciliations,
                expected_hand_count=expected_hand_count,
            )
        )
    except (LiveSelectionError, ValueError):
        consensus = None
        frame_candidates = ()
        select_evidence = []
        geometry_stable = False
    accepted = bool(
        absolutes.get("accepted") is True
        and geometry_stable
        and consensus is not None
        and consensus.accepted
        and consensus.selected_slot is None
        and len(select_evidence) == len(frames)
        and all(
            evidence.get("geometry_available") is True
            and evidence.get("geometry_ambiguous") is False
            and evidence.get("geometry_raised_slots") == []
            and evidence.get("resolved_slots") == []
            for evidence in select_evidence
        )
    )
    return {
        "accepted": accepted,
        "stamina": absolutes.get("stamina") if accepted else None,
        "shield": absolutes.get("shield") if accepted else None,
        "confidence": absolutes.get("confidence", 0.0) if accepted else 0.0,
        "reason": (
            "live_resource_baseline_unselected_unanimous"
            if accepted
            else "live_resource_baseline_unresolved"
        ),
        "frame_ids": frame_ids,
        "candidate_boxes": [
            [list(candidate.box) for candidate in candidates]
            for candidates in frame_candidates
        ],
        "candidate_fragment_reconciliations": fragment_reconciliations,
        "geometry_stable": geometry_stable,
        "select_consensus": consensus.to_dict() if consensus is not None else None,
        "select_evidence": select_evidence,
        "unselected_geometry_authorized": accepted,
        "resource_observations": absolutes.get("observations", []),
    }


def _observe_recovered_selected_baseline(context, frames, *, complete_event, catalog, plan,
                                         selection_observation=None, cost_context=None):
    """Rebuild resources from a separately verified pending selection."""
    from live_decision.hif_gray_selection import (
        selected_zero_cost_baseline,
        ordinary_selected_resource_preview,
        recovered_selected_support_journal,
        ordinary_selected_zero_cost_baseline,
        recovered_selected_support_visible_id,
        recovered_selected_ordinary_plus_visible_id,
    )

    bundle = context.gkh_hif_opening_state
    journal = context.gkh_hif_gray_selected_source
    validation_journal = journal
    ordinary = journal.get('ordinary_safe_preview') is True
    action = journal.get('refresh_a', {}).get('proposal', {}).get(
        'candidate_traces', {}).get('high_score', {}).get('action', {})
    slot = action.get('source_slot') if ordinary else journal.get('permit', {}).get('slot')
    if type(slot) is not int or not 0 <= slot < len(bundle.state.deck.hand):
        raise LiveSelectionError('selected_recovery_target_unbound')
    consensus, rows, select_evidence, stable = (selection_observation if selection_observation is not None
        else _observe_live_selected_slot_consensus(context, frames, expected_selected_slot=slot,
            expected_hand_count=len(bundle.state.deck.hand)))
    frame_ids = tuple(frame_identifier(frame) for frame in frames)
    ref = bundle.state.deck.hand[slot]
    definition = bundle.kernel.cards[ref.card_id]
    card_id = definition.canonical_card_id or definition.card_id
    evidence = dict(frame_ids=frame_ids, geometry_stable=stable,
        select_consensus=consensus.to_dict() if consensus is not None else None,
        select_evidence=select_evidence,
        candidate_boxes=[[list(candidate.box) for candidate in row] for row in rows])
    if stable and consensus is not None and consensus.accepted and consensus.selected_slot == slot:
        resource_journal = journal
        visible_card_id = card_id
        if ordinary:
            from live_decision.hif_skill_support import observe_selected_skill_support

            try:
                support_visible_id = recovered_selected_support_visible_id(bundle,
                    complete_event=complete_event, journal=journal)
            except ValueError as error:
                raise LiveSelectionError(str(error), events=(evidence,)) from error
            if support_visible_id is not None:
                evidence['selected_skill_support'] = observe_selected_skill_support(
                    context, frames, bundle=bundle, selected_slot=slot,
                    frame_identifier=frame_identifier, selection_evidence=dict(
                        frame_ids=list(frame_ids), geometry_stable=stable,
                        select_consensus=evidence['select_consensus'],
                        select_evidence=select_evidence))
                try:
                    resource_journal = recovered_selected_support_journal(bundle,
                        complete_event=complete_event, journal=journal, evidence=evidence)
                except ValueError as error:
                    raise LiveSelectionError(str(error), events=(evidence,)) from error
                visible_card_id = support_visible_id
            else:
                from live_decision.hif_single_plus import observe_selected_single_plus

                projected_id = recovered_selected_ordinary_plus_visible_id(bundle,
                    complete_event=complete_event, journal=journal)
                discover_projection = (projected_id is None
                    and journal.get('single_plus_estimate') is None
                    and not ref.upgraded and bool(definition.upgrade_card_id))
                if discover_projection:
                    evidence['detail_identity'] = _hif_selected_detail_identity(
                        context, frames, (card_id,), catalog, plan)
                    discover_projection = evidence['detail_identity'].get('accepted') is not True
                if projected_id is not None or discover_projection:
                    receipt = observe_selected_single_plus(context, frames, bundle=bundle,
                        selected_slot=slot, frame_identifier=frame_identifier, selection_evidence=dict(
                            frame_ids=list(frame_ids), geometry_stable=stable,
                            select_consensus=evidence['select_consensus'], select_evidence=select_evidence))
                    if projected_id is not None and (receipt.get('accepted') is not True
                            or receipt.get('ordinary_plus_projection') is not True
                            or receipt.get('projected_card_id') != projected_id):
                        raise LiveSelectionError('selected_recovery_ordinary_plus_title_changed', events=(receipt,))
                    if receipt.get('accepted') is True and receipt.get('ordinary_plus_projection') is True:
                        evidence['single_plus_estimate'] = receipt
                        resource_journal = dict(journal, single_plus_estimate=receipt)
                        # The original selection may have stopped before its
                        # title was read. Revalidate this fresh receipt against
                        # the original complete state, then all normal identity
                        # and resource gates below, before retaining it.
                        visible_card_id = recovered_selected_ordinary_plus_visible_id(bundle,
                            complete_event=complete_event, journal=resource_journal)
                        if discover_projection:
                            validation_journal = resource_journal
        rank_preview = (ordinary_selected_resource_preview(bundle,
            complete_event=complete_event, journal=resource_journal) if ordinary else None)
        from live_decision.hif_rank_preview import bind_rank_preview

        context.gkh_hif_rank_preview_binding = bind_rank_preview(
            bundle, rank_preview, complete_event['core']['fields'][5])
        if (evidence.get('detail_identity', {}).get('accepted') is not True
                or evidence['detail_identity'].get('card_id') != visible_card_id):
            evidence['detail_identity'] = _hif_selected_detail_identity(
                context, frames, (visible_card_id,), catalog, plan)
        evidence['resource_transaction'] = _observe_selected_resource_transaction(
            context, frames, rows, frame_ids=frame_ids, selected_slot=slot,
            baseline=dict(accepted=True, stamina=bundle.state.stamina, shield=bundle.state.genki),
            card_effect_preview=rank_preview, cost_context=cost_context)
    try:
        validator = ordinary_selected_zero_cost_baseline if ordinary else selected_zero_cost_baseline
        opening_args = (dict(opening_run=context.gkh_hif_live_context,
            round_context=getattr(context, 'gkh_hif_round_context', None),
            data_root=CATALOG_PATH.parent)
            if ordinary and complete_event.get('event') == 'live_opening_state' else {})
        baseline = validator(bundle, complete_event=complete_event,
            journal=validation_journal, evidence=evidence,
            geometry_comparator=_same_live_candidate_geometry, **opening_args)
        if baseline.get('accepted') is True and validation_journal is not journal:
            context.gkh_hif_gray_selected_source = validation_journal
        return baseline
    except ValueError as error:
        raise LiveSelectionError(str(error), events=(evidence,)) from error


def _resolve_resource_delta_consensus(
    observations: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    *,
    baseline_value: int,
    preview_value: int,
) -> dict[str, Any]:
    """Resolve a signed delta from full words or independently clear frames."""

    accepted_values = tuple(
        observation.get("value")
        for observation in observations
        if observation.get("accepted") is True
        and type(observation.get("value")) is int
    )
    if (
        len(accepted_values) == 3
        and accepted_values[0] == accepted_values[1] == accepted_values[2]
        and baseline_value + accepted_values[0] == preview_value
    ):
        return {
            "accepted": True,
            "value": accepted_values[0],
            "confidence": min(
                float(observation.get("confidence", 0.0))
                for observation in observations
            ),
            "reason": "resource_delta_unanimous",
            "observations": [dict(observation) for observation in observations],
        }
    clean_zero = bool(
        baseline_value == preview_value
        and all(
            observation.get("accepted") is False
            and isinstance(observation.get("red_evidence"), dict)
            and observation["red_evidence"].get("raw_red_pixel_count") == 0
            and observation["red_evidence"].get("red_present") is False
            for observation in observations
        )
    )
    from live_decision.resource_delta_evidence import corroborated_speck_zero

    speck_zero = (not clean_zero and baseline_value == preview_value
        and corroborated_speck_zero(observations))
    zero = clean_zero or speck_zero
    return {
        "accepted": zero,
        "value": 0 if zero else None,
        "confidence": 1.0 if zero else 0.0,
        "reason": (
            "resource_delta_explicit_clean_zero"
            if clean_zero
            else "resource_delta_corroborated_subcomponent_zero" if speck_zero
            else "resource_delta_unresolved_or_nonconserving"
        ),
        "observations": [dict(observation) for observation in observations],
    }


def _observe_selected_resource_transaction(
    context: Context,
    frames: tuple[Any, Any, Any],
    frame_candidates: tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]],
    *,
    frame_ids: tuple[str, str, str],
    selected_slot: int | None,
    baseline: dict[str, Any],
    authoritative_evidence: tuple[dict[str, Any], ...] | None = None,
    card_effect_preview: dict[str, Any] | None = None,
    cost_context: Any | None = None,
) -> dict[str, Any]:
    """Bind card-face cost and both HUD deltas to one selected-card transaction."""

    invalid = {
        "accepted": False,
        "displayed_effective_cost": None,
        "stamina_delta": None,
        "shield_delta": None,
        "normalized_stamina": None,
        "normalized_shield": None,
        "confidence": 0.0,
        "reason": "selected_resource_transaction_not_applicable",
        "selected_slot": selected_slot,
        "frame_ids": list(frame_ids),
        "observations": {},
    }
    if (
        selected_slot is None
        or len(frames) != 3
        or len(frame_candidates) != 3
        or len(frame_ids) != 3
        or baseline.get("accepted") is not True
        or type(baseline.get("stamina")) is not int
        or type(baseline.get("shield")) is not int
    ):
        return invalid
    selected_candidates: list[Any] = []
    for candidates in frame_candidates:
        matches = [candidate for candidate in candidates if candidate.slot == selected_slot]
        if len(matches) != 1:
            invalid["reason"] = "selected_resource_transaction_candidate_unbound"
            return invalid
        selected_candidates.append(matches[0])

    absolutes = _observe_live_resource_absolutes(
        context,
        frames,
        authoritative_evidence=authoritative_evidence,
    )
    cost_observations = []
    for index, (frame, candidate) in enumerate(zip(frames, selected_candidates, strict=True)):
        evidence = (authoritative_evidence[index]
            if authoritative_evidence is not None and len(authoritative_evidence) == 3 else {})
        early_cost = evidence.get('selected_effective_cost', {})
        if (early_cost.get('slot') == selected_slot
                and _authoritative_live_ocr_bundle_bound(evidence,
                    frame_id=frame_ids[index], candidates=frame_candidates[index])):
            cost_observations.append(dict(early_cost['observation']))
        else:
            cost_observations.append(_ocr_selected_effective_cost(
                context if cost_context is None else cost_context, frame, candidate))
    cost_observations = tuple(cost_observations)
    if authoritative_evidence is not None and len(authoritative_evidence) == 3:
        stamina_delta_observations = tuple(
            dict(evidence.get("delta_observations", {}).get("stamina", {}))
            for evidence in authoritative_evidence
        )
        shield_delta_observations = tuple(
            dict(evidence.get("delta_observations", {}).get("shield", {}))
            for evidence in authoritative_evidence
        )
    else:
        stamina_delta_observations = tuple(
            _ocr_resource_preview_delta(
                context,
                frame,
                "stamina",
                LIVE_STAMINA_PREVIEW_DELTA_ROI,
            )
            for frame in frames
        )
        shield_delta_observations = tuple(
            _ocr_resource_preview_delta(
                context,
                frame,
                "shield",
                LIVE_SHIELD_PREVIEW_DELTA_ROI,
            )
            for frame in frames
        )
    cost_values = tuple(
        observation.get("value")
        for observation in cost_observations
        if observation.get("accepted") is True
        and type(observation.get("value")) is int
    )
    effective_cost = (
        cost_values[0]
        if len(cost_values) == 3 and cost_values[0] == cost_values[1] == cost_values[2]
        else None
    )
    preview_stamina = absolutes.get("stamina")
    preview_shield = absolutes.get("shield")
    if (
        absolutes.get("accepted") is not True
        or type(preview_stamina) is not int
        or type(preview_shield) is not int
    ):
        stamina_delta = {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": "resource_delta_absolute_unresolved",
            "observations": [dict(item) for item in stamina_delta_observations],
        }
        shield_delta = {
            "accepted": False,
            "value": None,
            "confidence": 0.0,
            "reason": "resource_delta_absolute_unresolved",
            "observations": [dict(item) for item in shield_delta_observations],
        }
    else:
        stamina_delta = _resolve_resource_delta_consensus(
            stamina_delta_observations,
            baseline_value=baseline["stamina"],
            preview_value=preview_stamina,
        )
        shield_delta = _resolve_resource_delta_consensus(
            shield_delta_observations,
            baseline_value=baseline["shield"],
            preview_value=preview_shield,
        )
    stamina_delta_value = stamina_delta.get("value")
    shield_delta_value = shield_delta.get("value")
    expected_shield_spend = (
        min(baseline["shield"], effective_cost)
        if type(effective_cost) is int
        else None
    )
    expected_shield_delta = (
        -expected_shield_spend if type(expected_shield_spend) is int else None
    )
    expected_stamina_delta = (
        -(effective_cost - expected_shield_spend)
        if type(effective_cost) is int and type(expected_shield_spend) is int
        else None
    )
    conservation = {
        "stamina": bool(
            stamina_delta.get("accepted") is True
            and baseline["stamina"] + stamina_delta_value == preview_stamina
        ),
        "shield": bool(
            shield_delta.get("accepted") is True
            and baseline["shield"] + shield_delta_value == preview_shield
        ),
        "effective_cost": bool(
            type(effective_cost) is int
            and type(stamina_delta_value) is int
            and type(shield_delta_value) is int
            and effective_cost == -(stamina_delta_value + shield_delta_value)
        ),
        "shield_first_split": bool(
            type(expected_shield_delta) is int
            and type(expected_stamina_delta) is int
            and shield_delta_value == expected_shield_delta
            and stamina_delta_value == expected_stamina_delta
        ),
    }
    if card_effect_preview is not None:
        contract = card_effect_preview
        model_bound = (isinstance(contract, dict)
            and all(type(contract.get(name)) is int and contract[name] >= 0 for name in (
                'selected_slot', 'baseline_stamina', 'baseline_shield',
                'effective_cost', 'preview_stamina', 'preview_shield'))
            and contract.get('source') == 'native_hif_card_effect_preview'
            and contract.get('selected_slot') == selected_slot
            and contract.get('baseline_stamina') == baseline['stamina']
            and contract.get('baseline_shield') == baseline['shield'])
        if not model_bound:
            return {**invalid, 'reason': 'selected_resource_preview_contract_invalid'}
        displayed_cost = contract.get('displayed_card_cost', contract['effective_cost'])
        if (type(displayed_cost) is not int or displayed_cost < 0
                or ('displayed_card_cost' in contract and (
                    type(contract.get('nullify_cost_cards')) is not int
                    or contract['nullify_cost_cards'] <= 0 or contract['effective_cost'] != 0))):
            return {**invalid, 'reason': 'selected_resource_preview_displayed_cost_invalid'}
        conservation = {name:conservation[name] for name in ('stamina','shield')}
        conservation.update(declared_cost=model_bound and type(effective_cost) is int
            and effective_cost == displayed_cost,
            modeled_card_preview=model_bound and preview_stamina == contract.get('preview_stamina')
            and preview_shield == contract.get('preview_shield'))
        expected_stamina_delta = contract.get('preview_stamina',-1)-baseline['stamina']
        expected_shield_delta = contract.get('preview_shield',-1)-baseline['shield']
    accepted = bool(
        type(effective_cost) is int
        and stamina_delta.get("accepted") is True
        and shield_delta.get("accepted") is True
        and all(conservation.values())
    )
    confidence_values = [
        float(observation.get("confidence", 0.0))
        for observation in cost_observations
    ]
    effective_cost_confidence = min(confidence_values) if confidence_values else 0.0
    absolute_confidence = (
        float(absolutes.get("confidence", 0.0))
        if absolutes.get("accepted") is True
        else 0.0
    )
    preview_delta_confidence = (
        min(
            float(stamina_delta.get("confidence", 0.0)),
            float(shield_delta.get("confidence", 0.0)),
        )
        if stamina_delta.get("accepted") is True
        and shield_delta.get("accepted") is True
        else 0.0
    )
    if absolutes.get("accepted") is True:
        confidence_values.append(absolute_confidence)
    for delta in (stamina_delta, shield_delta):
        if delta.get("accepted") is True and delta.get("value") != 0:
            confidence_values.append(float(delta.get("confidence", 0.0)))
    if card_effect_preview is not None:
        # Model-bound payment was independently checked above. Resource
        # baselines and normalized HUD fields must not inherit auxiliary
        # printed-cost OCR confidence through this shared aggregate.
        confidence_values = [absolute_confidence, preview_delta_confidence]
    return {
        "accepted": accepted,
        "displayed_effective_cost": effective_cost if accepted else None,
        "card_face_cost": effective_cost if accepted else None,
        "effective_payment_cost": (
            card_effect_preview['effective_cost'] if accepted and card_effect_preview is not None
            else effective_cost if accepted else None),
        "stamina_delta": stamina_delta_value if accepted else None,
        "shield_delta": shield_delta_value if accepted else None,
        "preview_stamina": preview_stamina if absolutes.get("accepted") is True else None,
        "preview_shield": preview_shield if absolutes.get("accepted") is True else None,
        "normalized_stamina": baseline["stamina"] if accepted else None,
        "normalized_shield": baseline["shield"] if accepted else None,
        "confidence": min(confidence_values) if accepted and confidence_values else 0.0,
        "displayed_effective_cost_confidence": effective_cost_confidence,
        "preview_delta_confidence": preview_delta_confidence,
        "resource_absolute_confidence": absolute_confidence,
        "reason": (
            "selected_resource_transaction_conserved"
            if accepted
            else "selected_resource_transaction_unresolved_or_nonconserving"
        ),
        "selected_slot": selected_slot,
        "frame_ids": list(frame_ids),
        "candidate_boxes": [list(candidate.box) for candidate in selected_candidates],
        "expected_stamina_delta": expected_stamina_delta,
        "expected_shield_delta": expected_shield_delta,
        "conservation": conservation,
        "observations": {
            "absolutes": absolutes,
            "effective_cost": [dict(item) for item in cost_observations],
            "stamina_delta": stamina_delta,
            "shield_delta": shield_delta,
        },
        "card_effect_preview": card_effect_preview,
    }


def _observe_unselected_resource_transaction(
    context: Context,
    frames: tuple[Any, Any, Any],
    *,
    frame_ids: tuple[str, str, str],
    baseline: dict[str, Any],
    authoritative_evidence: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Revalidate an unselected refresh against the pre-input baseline."""

    absolutes = _observe_live_resource_absolutes(
        context,
        frames,
        authoritative_evidence=authoritative_evidence,
    )
    preview_stamina = absolutes.get("stamina")
    preview_shield = absolutes.get("shield")
    valid_baseline = bool(
        baseline.get("accepted") is True
        and type(baseline.get("stamina")) is int
        and type(baseline.get("shield")) is int
        and absolutes.get("accepted") is True
        and type(preview_stamina) is int
        and type(preview_shield) is int
    )
    if authoritative_evidence is not None and len(authoritative_evidence) == 3:
        stamina_observations = tuple(
            dict(evidence.get("delta_observations", {}).get("stamina", {}))
            for evidence in authoritative_evidence
        )
        shield_observations = tuple(
            dict(evidence.get("delta_observations", {}).get("shield", {}))
            for evidence in authoritative_evidence
        )
    else:
        stamina_observations = tuple(
            _ocr_resource_preview_delta(
                context,
                frame,
                "stamina",
                LIVE_STAMINA_PREVIEW_DELTA_ROI,
            )
            for frame in frames
        )
        shield_observations = tuple(
            _ocr_resource_preview_delta(
                context,
                frame,
                "shield",
                LIVE_SHIELD_PREVIEW_DELTA_ROI,
            )
            for frame in frames
        )
    if valid_baseline:
        stamina_delta = _resolve_resource_delta_consensus(
            stamina_observations,
            baseline_value=baseline["stamina"],
            preview_value=preview_stamina,
        )
        shield_delta = _resolve_resource_delta_consensus(
            shield_observations,
            baseline_value=baseline["shield"],
            preview_value=preview_shield,
        )
    else:
        stamina_delta = {"accepted": False, "value": None, "confidence": 0.0}
        shield_delta = {"accepted": False, "value": None, "confidence": 0.0}
    accepted = bool(
        valid_baseline
        and stamina_delta.get("accepted") is True
        and stamina_delta.get("value") == 0
        and shield_delta.get("accepted") is True
        and shield_delta.get("value") == 0
    )
    return {
        "accepted": accepted,
        "displayed_effective_cost": None,
        "stamina_delta": 0 if accepted else None,
        "shield_delta": 0 if accepted else None,
        "preview_stamina": preview_stamina if valid_baseline else None,
        "preview_shield": preview_shield if valid_baseline else None,
        "normalized_stamina": baseline.get("stamina") if accepted else None,
        "normalized_shield": baseline.get("shield") if accepted else None,
        "confidence": float(absolutes.get("confidence", 0.0)) if accepted else 0.0,
        "displayed_effective_cost_confidence": None,
        "preview_delta_confidence": (
            min(
                float(stamina_delta.get("confidence", 0.0)),
                float(shield_delta.get("confidence", 0.0)),
            )
            if accepted
            else 0.0
        ),
        "resource_absolute_confidence": (
            float(absolutes.get("confidence", 0.0)) if accepted else 0.0
        ),
        "reason": (
            "unselected_resource_transaction_conserved"
            if accepted
            else "unselected_resource_transaction_unresolved"
        ),
        "selected_slot": None,
        "frame_ids": list(frame_ids),
        "conservation": {
            "stamina": stamina_delta.get("accepted") is True,
            "shield": shield_delta.get("accepted") is True,
            "effective_cost": True,
        },
        "observations": {
            "absolutes": absolutes,
            "stamina_delta": stamina_delta,
            "shield_delta": shield_delta,
        },
    }


def _live_generation_fingerprint(
    fields: dict[str, Any],
    *,
    hand_count: int,
    plan: str,
) -> dict[str, Any]:
    names = (
        "turn",
        "turns_remaining",
        "stamina",
        "shield",
        "score",
        "score_multiplier",
    )
    if any(not fields.get(name, {}).get("accepted") for name in names):
        raise LiveSelectionError("Live generation fingerprint fields are incomplete")
    action_order = fields.get("action_order", {})
    if not action_order.get("accepted"):
        raise LiveSelectionError("Live generation action order is incomplete")
    return {
        "turn": fields["turn"]["value"],
        "turns_remaining": fields["turns_remaining"]["value"],
        "stamina": fields["stamina"]["value"],
        "shield": fields["shield"]["value"],
        "score": fields["score"]["value"],
        "score_multiplier": fields["score_multiplier"]["value"],
        "action_order": list(action_order["value"]),
        "hand_count": hand_count,
        "plan": plan,
    }


def _current_live_identity_confidence(certificate, fresh, *, event, selected_slot, bound_hif,
                                      minimum_confidence, current_title_confidence=None):
    """Keep source and current scores separate after an independent B identity.

    Earlier identity still supplies its verified normal/plus member. A current
    strong exact-family observation supplies its own raw score; a carried or
    weak observation cannot become strong merely by retaining that member.
    """
    current = float(current_title_confidence if current_title_confidence is not None else fresh.confidence)
    original = min(float(certificate['probe_confidence']), current)
    checks = [p for p in event.get('identity_revalidation', ()) if p.get('slot') == fresh.slot]
    if (bound_hif and event.get('phase') == 'authoritative_refresh_b'
            and fresh.slot == selected_slot and fresh.accepted
            and len(checks) == 1 and checks[0].get('accepted') is True
            and checks[0].get('acceptance_mode') == 'fresh_exact_family'
            and checks[0].get('carried') is False and checks[0].get('standard_accepted') is True
            and fresh.frame_id == event['frame_ids'][-1]
            and type(minimum_confidence) in (int, float)
            and original < minimum_confidence <= current <= 1):
        event.setdefault('current_identity_scores', []).append(dict(slot=fresh.slot,
            frame_id=fresh.frame_id, original_probe_confidence=certificate['probe_confidence'],
            current_confidence=current, raw_visual_confidence=fresh.confidence,
            source='independent_current_exact_family' if current_title_confidence is None else 'independent_current_full_title'))
        return current
    return original


def _build_live_identity_certificate(
    card_predictions: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if not card_predictions:
        raise LiveSelectionError("Live identity certificate is empty")
    from live_decision.hif_bound_title import family as identity_family

    certificate = []
    for slot, entry in enumerate(card_predictions):
        prediction = entry["prediction"]
        provenance = entry.get('identity_provenance')
        provenance = provenance if isinstance(provenance, dict) else {}
        family = identity_family(prediction, provenance.get('bound_title_evidence') or provenance.get('inspected_art_evidence'))
        if (
            prediction.slot != slot
            or not prediction.accepted
            or prediction.card_id == UNKNOWN
            or not family
            or prediction.card_id not in family
            or type(entry.get("upgraded")) is not bool
        ):
            raise LiveSelectionError("Live identity certificate is incomplete")
        certificate.append(
            {
                "slot": slot,
                "family": family,
                "card_id": prediction.card_id,
                "upgraded": entry["upgraded"],
                "probe_confidence": prediction.confidence,
                "probe_margin": prediction.margin,
                "upgrade_evidence": entry.get("upgrade_evidence"),
                "source_prediction": prediction,
                "source_selected_slot": (
                    entry.get("identity_provenance", {}).get("source_selected_slot")
                    if isinstance(entry.get("identity_provenance"), dict)
                    else None
                ),
                "unselected_view_verified": bool(
                    isinstance(entry.get("identity_provenance"), dict)
                    and entry["identity_provenance"].get(
                        "unselected_view_verified"
                    )
                    is True
                ),
                "selected_transition_verified": bool(
                    isinstance(entry.get("identity_provenance"), dict)
                    and entry["identity_provenance"].get(
                        "selected_transition_verified"
                    )
                    is True
                ),
            }
        )
        provenance = entry.get('identity_provenance')
        if isinstance(provenance, dict) and provenance.get('inspected_art_evidence') is not None:
            certificate[-1]['inspected_art_evidence'] = provenance['inspected_art_evidence']
        if isinstance(provenance, dict) and provenance.get('bound_title_evidence') is not None:
            certificate[-1]['bound_title_evidence'] = provenance['bound_title_evidence']
        if isinstance(provenance, dict) and provenance.get('selected_neighbor_identity_carry') is not None:
            certificate[-1]['selected_neighbor_identity_carry'] = provenance['selected_neighbor_identity_carry']
    return tuple(certificate)


class _LiveCaptureTimingError(RuntimeError):
    """Internal bounded-clock failure converted to an audited selection stop."""

    def __init__(self, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


def _read_strict_live_capture_clock() -> float:
    try:
        observed = time.perf_counter()
    except Exception as error:
        raise _LiveCaptureTimingError(
            "authoritative_live_capture_clock_invalid",
            timing_failure="perf_counter_read_failed",
            clock_error_type=type(error).__name__,
        ) from error
    if (
        type(observed) not in {int, float}
        or not math.isfinite(float(observed))
        or float(observed) < 0.0
    ):
        raise _LiveCaptureTimingError(
            "authoritative_live_capture_clock_invalid",
            timing_failure="perf_counter_not_finite_nonnegative_number",
            clock_value_type=type(observed).__name__,
        )
    return float(observed)


def _wait_until_live_capture_not_before(
    not_before: float,
    *,
    previous_capture_started_at: float | None = None,
) -> None:
    """Bound capture-start spacing to at most two finite-clock sleeps."""

    if (
        type(not_before) not in {int, float}
        or not math.isfinite(float(not_before))
        or float(not_before) < 0.0
        or (
            previous_capture_started_at is not None
            and (
                type(previous_capture_started_at) not in {int, float}
                or not math.isfinite(float(previous_capture_started_at))
                or float(previous_capture_started_at) < 0.0
                or float(not_before) < float(previous_capture_started_at)
            )
        )
    ):
        raise _LiveCaptureTimingError(
            "authoritative_live_capture_clock_invalid",
            timing_failure="capture_schedule_not_finite_monotonic",
        )

    scheduled_at = float(not_before)
    previous_capture = (
        float(previous_capture_started_at)
        if previous_capture_started_at is not None
        else None
    )
    observed = _read_strict_live_capture_clock()
    if previous_capture is not None and observed < previous_capture:
        raise _LiveCaptureTimingError(
            "authoritative_live_capture_clock_invalid",
            timing_failure="clock_moved_backward_before_interval_wait",
            sleep_count=0,
        )

    sleep_count = 0
    while observed < scheduled_at and sleep_count < 2:
        try:
            time.sleep(scheduled_at - observed)
        except Exception as error:
            raise _LiveCaptureTimingError(
                "authoritative_live_capture_clock_invalid",
                timing_failure="capture_interval_sleep_failed",
                sleep_count=sleep_count,
                sleep_error_type=type(error).__name__,
            ) from error
        sleep_count += 1
        current = _read_strict_live_capture_clock()
        if current <= observed:
            raise _LiveCaptureTimingError(
                "authoritative_live_capture_clock_invalid",
                timing_failure=(
                    "clock_stalled_during_interval_wait"
                    if current == observed
                    else "clock_moved_backward_during_interval_wait"
                ),
                sleep_count=sleep_count,
            )
        observed = current

    if observed < scheduled_at:
        raise _LiveCaptureTimingError(
            "authoritative_live_capture_interval_unverified",
            timing_failure="capture_interval_sleep_budget_exhausted",
            sleep_count=sleep_count,
        )


def _unresolved_select_label_diagnostic(event, slot):
    """Validate retry-only evidence; plate spelling never identifies SELECT.

    1669 also read ELEC after 1668's SELEC. Enumerating fragments mistakes a
    request for one new observation for an identity proof. All acceptance
    remains with the unchanged fresh exact OCR or complete glyph proof.
    """
    return bool(isinstance(event, dict) and type(slot) is int
        and event.get('phase') == 'hif_select_label' and event.get('reason') == 'select_label_accent_unresolved'
        and event.get('accepted') is False
        and type(event.get('input_click_count')) is int and event['input_click_count'] == 0
        and event.get('slot') == slot and event.get('raised_slots') == [slot]
        and isinstance(event.get('frame_id'), str) and bool(event['frame_id'])
        and all(isinstance(event.get(name), dict)
            and _valid_ocr_result(SimpleNamespace(**event[name]))
            and event[name].get('box') == [0,0,96,48]
            and isinstance(event[name].get('text'), str) for name in ('original_ocr', 'plate_ocr'))
        # Confidence in a rejected spelling cannot disqualify an otherwise
        # identical request for one fresh batch. This never accepts the text.
        and event['original_ocr']['text'] == 'SÉLECT' and 0 < event['original_ocr']['score'] <= 1
        # Exact plate text keeps its existing strong acceptance / weak retry.
        and event['plate_ocr']['text'].strip() != 'SELECT')


def _selected_accent_label_reobserve_allowed(context, event, kwargs):
    """Spend only the calling A/B stage's existing retry, never prove SELECT."""
    baseline = kwargs.get('resource_baseline') or {}
    journal = getattr(context, 'gkh_hif_gray_selected_source', None)
    bundle = getattr(context, 'gkh_hif_opening_state', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    counts = kwargs.get('input_click_counts')
    slot = kwargs.get('expected_selected_slot')
    frames = event.get('capture_frame_ids', ())
    source = journal.get('permit', {}) if isinstance(journal, dict) else {}
    source_slot = source.get('slot')
    if isinstance(journal, dict) and journal.get('ordinary_safe_preview') is True:
        # The existing selected-source loader verifies this ordinary safe A;
        # it has no gray permit. Bind the same recorded slot/instance and the
        # complete run, solely to spend the existing stage's observation retry.
        a = journal.get('refresh_a', {})
        action = a.get('proposal', {}).get('candidate_traces', {}).get('high_score', {}).get('action', {})
        safe = a.get('safe_decision', {})
        source = getattr(context, 'gkh_hif_committed_event', None)
        source_slot = action.get('source_slot')
        if (not isinstance(source, dict) or source.get('accepted') is not True
                or source.get('event') != 'live_committed_state' or source.get('strategy_state_bound') is not True
                or not source.get('checks') or not all(value is True for value in source['checks'].values())
                or bundle is None or type(source_slot) is not int
                or not 0 <= source_slot < len(bundle.state.deck.hand)
                or 'permit' in journal or a.get('accepted') is not True
                or a.get('expected_selected_slot') is not None or a.get('unselected_geometry_authorized') is not True
                or safe.get('status') != 'ALLOW' or safe.get('has_action_intent') is not True
                or safe.get('action') != dict(kind='PLAY_CARD', card_slot=source_slot)
                or action.get('kind') != 'PLAY_CARD'
                or action.get('card_instance_id') != bundle.state.deck.hand[source_slot].instance_id):
            return False
    return bool(isinstance(journal, dict)
        and bundle is not None and isinstance(run, HifLiveRunContext)
        and (bundle.run_id, bundle.revision) == (run.run_id, run.revision)
            == (source.get('run_id'), source.get('revision'))
        and journal.get('selection_ready') is True
        and type(journal.get('selection_count')) is int and journal['selection_count'] == 1
        and journal.get('business_submit_count', 0) == 0 and source_slot == slot
        and baseline.get('accepted') is True and baseline.get('selected_geometry_authorized') is True
        and baseline.get('selected_slot') == slot
        and isinstance(counts, tuple) and len(counts) == 3
        and all(type(count) is int and count == 0 for count in counts)
        and _unresolved_select_label_diagnostic(event, slot)
        and kwargs.get('phase') in ('authoritative_refresh_a', 'authoritative_refresh_b')
        and event.get('capture_phase') == kwargs['phase']
        and type(event.get('select_retries_remaining')) is int and event['select_retries_remaining'] > 0
        and type(event.get('capture_retry_index')) is int and event['capture_retry_index'] == 0
        and 1 <= len(frames) <= 3 and len(set(frames)) == len(frames)
        and all(isinstance(frame, str) and frame for frame in frames)
        and event.get('frame_id') == frames[-1]
        and not set(frames) & set((*baseline.get('frame_ids', ()),
            *journal.get('refresh_a', {}).get('frame_ids', ()), *journal.get('refresh_b', {}).get('frame_ids', ()))))


def _selected_mosaic_low_confidence_reobserve_allowed(event, kwargs):
    """Discard one weak conflicting batch, never accept either conflicting read."""
    stop = event.get('ocr_contradiction_stop') or {}
    common = stop.get('common') or {}
    conflict = stop.get('fallback_conflict') or {}
    name = conflict.get('field')
    weak, direct = conflict.get('mosaic_observation') or {}, conflict.get('direct_observation') or {}
    rows = (common.get('results') or {}).get(name, ())
    row_confidence = (rows[0].get('score') if len(rows) == 1 and isinstance(rows[0], dict)
                      else getattr(rows[0], 'score', None) if len(rows) == 1 else None)
    try:
        threshold = max(float(kwargs['params']['debug_resource_absolute_min_confidence']),
                        LIVE_AUTHORITATIVE_OCR_MOSAIC_ABSOLUTE_MIN_CONFIDENCE)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(event.get('reason') == 'authoritative_live_ocr_mosaic_contradiction'
        and event.get('phase') == kwargs.get('phase')
        and event.get('click_dispatched') is False and event.get('input_may_have_been_dispatched') is False
        and event.get('geometry_stable') is True and event.get('retry_index') == 0
        and event.get('expected_selected_slot') == kwargs.get('expected_selected_slot')
        and len(event.get('frame_ids', ())) == 3 and len(set(event['frame_ids'])) == 3
        and stop.get('reason') == 'authoritative_live_common_mosaic_contradiction'
        and common.get('accepted') is True and common.get('contradictions') == []
        and name in ('stamina', 'shield') and len(rows) == 1
        and weak.get('accepted') is True and direct.get('accepted') is True
        and type(weak.get('value')) is int and type(direct.get('value')) is int
        and weak['value'] == conflict.get('mosaic_value') != conflict.get('direct_value') == direct['value']
        and type(weak.get('confidence')) in (int,float) and type(direct.get('confidence')) in (int,float)
        and 0 <= weak['confidence'] < threshold <= direct['confidence'] <= 1
        and row_confidence == weak['confidence'])


def _selected_stale_refresh_reobserve_allowed(event, kwargs):
    """Only replace an otherwise complete expired selected A, never retime it."""
    from live_decision.hif_native.native_card_evidence import input_window_limit

    slot = kwargs.get('expected_selected_slot')
    frames = event.get('frame_ids', ())
    identities = event.get('identity_revalidation', ())
    selected = event.get('select_consensus') or {}
    observations = selected.get('observations')
    safe = event.get('safe_decision') or {}
    age = event.get('capture_age_ms')
    proposal = event.get('proposal') or {}
    return bool(kwargs.get('phase') == event.get('phase') == 'authoritative_refresh_a'
        and type(slot) is int and event.get('expected_selected_slot') == slot
        and event.get('accepted') is False and event.get('reason') == 'stale_state'
        and event.get('click_dispatched') is False and event.get('input_may_have_been_dispatched') is False
        and event.get('geometry_stable') is True and event.get('selected_identity_authorized') is True
        and selected.get('accepted') is True and selected.get('selected_slot') == slot
        and isinstance(observations, (tuple, list)) and len(observations) == 3
        and all(isinstance(row, (tuple, list)) and len(row) == 1
                and type(row[0]) is int and row[0] == slot for row in observations)
        and len(frames) == 3 and len(set(frames)) == 3
        and event.get('resource_transaction', {}).get('accepted') is True
        and len(identities) == kwargs.get('expected_hand_count')
        and all(row.get('accepted') is True for row in identities)
        and safe.get('has_action_intent') is False and safe.get('reason') == 'stale_state'
        and proposal.get('status') == 'PROPOSE'
        and proposal.get('action') == dict(kind='PLAY_CARD', card_slot=slot)
        and type(age) in (int, float) and math.isfinite(age)
        and age > input_window_limit(event, kwargs.get('params', {}).get('max_snapshot_age_ms', math.inf)))


def _capture_selected_live_refresh(context, identity_certificate, **kwargs):
    """Replace rejected selected batches within their existing A/B budgets."""
    prior = []
    retried_reasons = set()
    for attempt in range(3):
        try:
            result = _capture_verified_live_refresh(context, identity_certificate, **kwargs)
            if _selected_stale_refresh_reobserve_allowed(result['event'], kwargs):
                # A stale safety decision is returned, while the existing
                # selected-batch retry handles raised recognition failures.
                raise LiveSelectionError('stale_state', events=(result['event'],))
            if prior:
                result['event']['select_frame_reobservation'] = prior
            return result
        except LiveSelectionError as error:
            evidence = error.events
            if kwargs.get('phase') == 'authoritative_refresh_a':
                selected_journal = getattr(context, 'gkh_hif_gray_selected_source', None)
                bundle = getattr(context, 'gkh_hif_opening_state', None)
                baseline = kwargs.get('resource_baseline') or {}
                slot = kwargs.get('expected_selected_slot')
                source_slot = (selected_journal.get('refresh_a', {}).get('proposal', {}).get(
                    'candidate_traces', {}).get('high_score', {}).get('action', {}).get('source_slot')
                    if isinstance(selected_journal, dict)
                    and selected_journal.get('ordinary_safe_preview') is True
                    else selected_journal.get('permit', {}).get('slot')
                    if isinstance(selected_journal, dict) else None)
                counts = kwargs.get('input_click_counts')
                mosaic_noise = (len(evidence) == 1
                    and _selected_mosaic_low_confidence_reobserve_allowed(evidence[0], kwargs))
                stale_selected = (len(evidence) == 1
                    and _selected_stale_refresh_reobserve_allowed(evidence[0], kwargs))
                select_noise = (len(evidence) == 1 and evidence[0].get('phase') == 'hif_select_label'
                    and (evidence[0].get('reason') == 'exact_select_low_confidence'
                         or _selected_accent_label_reobserve_allowed(context, evidence[0], kwargs))
                    and evidence[0].get('slot') == slot
                    and type(evidence[0].get('input_click_count')) is int
                    and evidence[0]['input_click_count'] == 0
                    and isinstance(evidence[0].get('frame_id'), str) and bool(evidence[0]['frame_id']))
                retryable = (len(evidence) == 1
                    and isinstance(getattr(context, 'gkh_hif_live_context', None), HifLiveRunContext)
                    and bundle is not None and hasattr(bundle, 'state')
                    and type(slot) is int and type(source_slot) is int and source_slot == slot
                    and 0 <= slot < len(bundle.state.deck.hand) <= 5
                    and kwargs.get('expected_hand_count') == len(bundle.state.deck.hand)
                    and baseline.get('accepted') is True
                    and baseline.get('selected_geometry_authorized') is True
                    and baseline.get('selected_slot') == slot
                    and kwargs.get('expected_generation') is None
                    and kwargs.get('allow_input_free_snapshot_retry') is True
                    and type(kwargs.get('snapshot_retries_remaining')) is int
                    and kwargs['snapshot_retries_remaining'] == 1
                    and type(kwargs.get('retry_index', 0)) is int
                    and kwargs.get('retry_index', 0) == 0
                    and isinstance(counts, tuple) and len(counts) == 3
                    and all(type(count) is int and count == 0 for count in counts)
                    and (select_noise or mosaic_noise or stale_selected))
                if not retryable:
                    raise
                # This catches SELECT failures anywhere in capture, including
                # normalization before the aggregate refresh event exists.
                fresh_kwargs = dict(kwargs, select_retries_remaining=0,
                    hand_count_retries_remaining=0, allow_input_free_snapshot_retry=False,
                    snapshot_retries_remaining=0, retry_index=1)
                try:
                    result = _capture_verified_live_refresh(context, identity_certificate, **fresh_kwargs)
                except LiveSelectionError as fresh_error:
                    raise LiveSelectionError(str(fresh_error),
                        events=(*evidence, *fresh_error.events)) from fresh_error
                result['event'].update(snapshot_retry_count=1, snapshot_retries_remaining=0,
                    snapshot_retry_kind=('recovered_selected_stale_a_full_batch' if stale_selected
                                         else 'recovered_selected_weak_mosaic_full_batch' if mosaic_noise
                                         else 'recovered_selected_select_full_batch'),
                    prior_attempts=[*evidence, *result['event'].get('prior_attempts', [])])
                return result
            from live_decision.opening_observer import (
                selected_resource_reobserve_allowed,
                selected_refresh_noise_reobserve_allowed,
            )
            pair_layout_transient = False
            if (len(evidence)==1 and kwargs.get('expected_hand_count') in (2, 3, 4)
                    and evidence[0].get('reason')=='authoritative_live_ocr_mosaic_source_unavailable'):
                rows = evidence[0].get('candidate_boxes', ())
                valid = []
                for boxes in rows:
                    try:
                        if len(boxes)!=kwargs.get('expected_hand_count'):
                            raise ValueError('pair count changed')
                        hif_select_label_rois((1280,720,3),tuple(
                            SimpleNamespace(slot=i,box=box) for i,box in enumerate(boxes)))
                        valid.append(True)
                    except (ValueError,TypeError):
                        valid.append(False)
                pair_layout_transient = len(valid)==3 and valid.count(True)==2
            single_frame_missing = False
            if (len(evidence)==1 and kwargs.get('expected_hand_count')==1
                    and evidence[0].get('reason')=='authoritative_live_hand_count_changed'):
                rows=evidence[0].get('candidate_boxes',())
                complete=[row for row in rows if len(row)==1]
                single_frame_missing=(len(rows)==3 and len(complete)==2
                    and sum(not row for row in rows)==1
                    and all(len(row[0])==4 and abs(row[0][0]-265)<=8
                            and 852<=row[0][1]<=866 and row[0][2:]==[190,250]
                            for row in complete)
                    and abs(complete[0][0][1]-complete[1][0][1])<=2)
            supported_failure = (len(evidence)==1 and (
                (evidence[0].get('phase')=='authoritative_refresh_b'
                 and evidence[0].get('reason')=='authoritative_live_evidence_window_expired'
                 and evidence[0].get('expected_selected_slot')==kwargs.get('expected_selected_slot')
                 and len(evidence[0].get('candidate_boxes',()))==3
                 and all(len(row)==kwargs.get('expected_hand_count')
                         for row in evidence[0]['candidate_boxes'])
                 and all(len(box)==4 for row in evidence[0]['candidate_boxes'] for box in row)
                 and all(abs(a[0]-b[0])<=2 and abs(a[1]-b[1])<=2 and a[2:]==b[2:]
                         for row in evidence[0]['candidate_boxes'][1:]
                         for a,b in zip(evidence[0]['candidate_boxes'][0],row,strict=True)))
                or
                (evidence[0].get('phase')=='hif_select_label'
                 and evidence[0].get('reason')=='exact_select_low_confidence'
                 and evidence[0].get('slot')==kwargs.get('expected_selected_slot'))
                or _selected_accent_label_reobserve_allowed(context, evidence[0], kwargs)
                or _selected_mosaic_low_confidence_reobserve_allowed(evidence[0], kwargs)
                or pair_layout_transient
                or single_frame_missing
                or (evidence[0].get('reason') == 'authoritative_live_hand_count_changed'
                    and kwargs.get('expected_hand_count') in (2, 3, 4, 5)
                    and _hif_selected_five_frame_missing(
                        evidence[0].get('candidate_boxes', ()),
                        kwargs.get('expected_selected_slot'), kwargs.get('expected_hand_count')))
                or selected_resource_reobserve_allowed(evidence[0],kwargs.get('params',{}),
                    kwargs.get('expected_selected_slot'))
                or selected_refresh_noise_reobserve_allowed(evidence[0],kwargs.get('params',{}),
                    kwargs.get('expected_selected_slot'),kwargs.get('expected_hand_count'))))
            failure_reason = evidence[0].get('reason') if len(evidence) == 1 else None
            retry_reason = ('exact_select_low_confidence'
                if failure_reason == 'select_label_accent_unresolved' else failure_reason)
            retryable = (attempt < 2 and retry_reason not in retried_reasons
                and kwargs.get('phase') == 'authoritative_refresh_b'
                and kwargs.get('selected_transition_verified') is True
                and getattr(context, 'gkh_hif_opening_state', None) is not None
                and supported_failure)
            if not retryable:
                raise LiveSelectionError(str(error), events=(*prior, *evidence)) from error
            prior.extend(evidence)
            retried_reasons.add(retry_reason)
            if retry_reason == 'exact_select_low_confidence':
                kwargs = dict(kwargs, select_retries_remaining=0)
            time.sleep(.08)


def _hif_selected_five_frame_missing(rows, selected_slot, hand_count=5):
    """Allow one fresh batch after an isolated empty detector frame."""
    anchors = {2: (157, 374), 3: (48, 265, 482), 4: (18, 183, 347, 512),
               5: (18, 141.25, 264.5, 387.75, 511)}.get(hand_count)
    if anchors is None or len(rows) != 3 or selected_slot not in range(hand_count):
        return False
    complete = [row for row in rows if len(row) == hand_count]
    if len(complete) != 2 or sum(not row for row in rows) != 1:
        return False
    for row in complete:
        for slot, box in enumerate(row):
            if (len(box) != 4 or abs(box[0] - anchors[slot]) > 3
                    or box[2:] != [190, 250]
                    or not ((852 <= box[1] <= 866) if slot == selected_slot
                            else (880 <= box[1] <= 890))):
                return False
    return all(abs(a[1] - b[1]) <= 2 for a, b in zip(*complete, strict=True))


def _hif_selected_unresolved_layout_batch(rows, selected_slot, hand_count):
    """Request a fresh batch, never accept an unresolved detector layout."""
    if len(rows) != 3 or hand_count not in (3, 4, 5):
        return False
    complete, unresolved_nonempty = [], False
    for row in rows:
        if any(len(box) != 4 or any(type(value) is not int for value in box) for box in row):
            return False
        if _hif_selected_five_frame_missing([row, [], row], selected_slot, hand_count):
            complete.append(row)
            continue
        if not row:
            continue
        # Another physical row is a conflict, even if its anchors or selected
        # slot differ. Only uncalibrated detector fragments remain UNKNOWN.
        if all(box[2:] == [190, 250] for box in row):
            return False
        try:
            hif_select_label_rois((1280, 720, 3), tuple(
                SimpleNamespace(slot=i, box=box) for i, box in enumerate(row)))
        except (ValueError, TypeError, IndexError):
            unresolved_nonempty = True
        else:
            return False
    return bool(complete and unresolved_nonempty and all(
        abs(a[0]-b[0]) <= 2 and abs(a[1]-b[1]) <= 2 and a[2:] == b[2:]
        for row in complete[1:] for a, b in zip(complete[0], row, strict=True)))


def _capture_verified_live_refresh(
    context: Context,
    identity_certificate: tuple[dict[str, Any], ...],
    *,
    expected_selected_slot: int | None,
    expected_generation: dict[str, Any] | None,
    params: dict[str, Any],
    plan: str,
    phase: str,
    recognizer: Any,
    catalog: SkillCardFeatureCatalog,
    policy: WeightedPolicyProvider,
    safety: DeterministicLiveSafetyFilter,
    expected_hand_count: int | None = None,
    resource_baseline: dict[str, Any] | None = None,
    selected_transition_verified: bool = False,
    allow_selected_score_preview_change: bool = False,
    select_retries_remaining: int = 2,
    hand_count_retries_remaining: int = 1,
    allow_input_free_snapshot_retry: bool = False,
    snapshot_retries_remaining: int = 0,
    input_click_counts: tuple[int, int, int] | None = None,
    retry_index: int = 0,
    selected_identity_journal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild one complete input-authorizing state from three fresh frames."""

    bound_hif = getattr(context, 'gkh_hif_opening_state', None)
    from live_decision.hif_rank_preview import bind_rank_preview

    context.gkh_hif_rank_preview_binding = None
    if (expected_selected_slot is not None and bound_hif is not None
            and getattr(policy, 'bundle', None) is not None
            and policy.bundle.state == bound_hif.state
            and callable(getattr(policy, 'selected_resource_preview', None))):
        context.gkh_hif_rank_preview_binding = bind_rank_preview(bound_hif,
            policy.selected_resource_preview(expected_selected_slot), policy.action_order)
    bound_continuation=(isinstance(getattr(context,'gkh_hif_live_context',None),HifLiveRunContext)
        and bound_hif is not None and hasattr(bound_hif,'state')
        and 1<=len(bound_hif.state.deck.hand)<=5
        and expected_hand_count==len(bound_hif.state.deck.hand))
    cost_context = (_SelectedCostOcrContext(context)
                    if bound_continuation and expected_selected_slot is not None else context)
    defer_middle_popup = bound_continuation
    projected_capture_span_ms: float | None = None
    continuation_pair = (
        isinstance(getattr(context, 'gkh_hif_live_context', None), HifLiveRunContext)
        and bound_hif is not None and len(bound_hif.state.deck.hand) in (1,2)
        and expected_hand_count == len(bound_hif.state.deck.hand))
    if expected_hand_count is not None and (
        type(expected_hand_count) is not int
        or (expected_hand_count != 3 and not bound_continuation)
        or len(identity_certificate) != expected_hand_count
    ):
        raise LiveSelectionError(
            "authoritative_live_expected_hand_count_mismatch",
            events=(
                {
                    "phase": phase,
                    "accepted": False,
                    "reason": "authoritative_live_expected_hand_count_mismatch",
                    "expected_hand_count": expected_hand_count,
                    "certificate_hand_count": len(identity_certificate),
                    "click_dispatched": False,
                },
            ),
        )

    frames_buffer: list[Any] = []
    frame_capture_started_at: list[float] = []
    frame_candidates: list[tuple[Any, ...]] = []
    frame_ids: list[str | None] = [None, None, None]
    select_observations: list[tuple[int, ...] | None] = [None, None, None]
    select_evidence: list[dict[str, Any] | None] = [None, None, None]
    first_frame_analysis_error: Exception | None = None
    first_frame_ocr_bundle: dict[str, Any] | None = None
    first_ocr_prework_elapsed_ms: float | None = None
    first_ocr_scheduling_reason: str | None = None
    middle_frame_ocr_bundle: dict[str, Any] | None = None
    middle_ocr_projected_capture_span_ms: float | None = None
    middle_ocr_capture_reserve_ms: float | None = None
    health_votes = 0
    playable_votes = 0
    secondary_votes = 0
    popup_votes = 0
    prior_input_may_have_been_dispatched = phase == "authoritative_refresh_b"

    def stop_timing(reason: str, **details: Any) -> NoReturn:
        event: dict[str, Any] = {
            "phase": phase,
            "accepted": False,
            "reason": reason,
            "click_dispatched": False,
            "input_may_have_been_dispatched": (
                prior_input_may_have_been_dispatched
            ),
            "retry_index": retry_index,
            "captured_frame_count": len(frames_buffer),
        }
        if isinstance(cost_context, _SelectedCostOcrContext):
            event['selected_cost_ocr_reuse'] = cost_context.counts
        if frame_capture_started_at:
            event["frame_capture_started_at"] = list(frame_capture_started_at)
        event.update(details)
        raise LiveSelectionError(reason, events=(event,))

    raw_max_snapshot_age_ms = params.get("max_snapshot_age_ms")
    if (
        type(raw_max_snapshot_age_ms) not in {int, float}
        or not math.isfinite(float(raw_max_snapshot_age_ms))
        or float(raw_max_snapshot_age_ms) <= 0.0
    ):
        stop_timing(
            "authoritative_live_capture_clock_invalid",
            timing_failure="max_snapshot_age_not_finite_positive_number",
            timing_field="max_snapshot_age_ms",
            timing_value_type=type(raw_max_snapshot_age_ms).__name__,
        )
    max_snapshot_age_ms = float(raw_max_snapshot_age_ms)
    from live_decision.hif_native.native_card_evidence import selected_observation_window_ms

    observation_window_ms = (selected_observation_window_ms(max_snapshot_age_ms, expected_selected_slot,
        strong_unselected=phase == 'authoritative_refresh_a')
        if bound_continuation and params.get('hif_policy_profile') == 'hif-strong-reference-trial-v1'
        else max_snapshot_age_ms)
    input_window_ms = (observation_window_ms if phase in ('authoritative_refresh_a', 'authoritative_refresh_b')
        else max_snapshot_age_ms)
    if input_window_ms != max_snapshot_age_ms:
        from copy import copy

        # Keep every other safety check and the original caller's configuration.
        # The strong card path uses one bounded window before and after SELECT.
        safety = copy(safety)
        safety.config = replace(safety.config, max_snapshot_age_ms=input_window_ms)

    scene_batch = None
    if bound_continuation:
        from live_decision.hif_authoritative_scene import prepare_scene_batch
        scene_batch = prepare_scene_batch(context, _recognition_hit)
    split_middle_popup = (expected_selected_slot is not None and expected_hand_count in (4,5)
        and getattr(scene_batch, 'popup_parts_supported', False) is True)
    middle_popup_started = False
    middle_popup_completed = False
    middle_popup_projected_capture_span_ms = None
    middle_popup_suffix_projected_capture_span_ms = None
    next_capture_not_before: float | None = None
    for index in range(3):
        try:
            if next_capture_not_before is not None:
                _wait_until_live_capture_not_before(
                    next_capture_not_before,
                    previous_capture_started_at=frame_capture_started_at[-1],
                )
            capture_started_at = _read_strict_live_capture_clock()
        except _LiveCaptureTimingError as error:
            stop_timing(
                error.reason,
                frame_index=index,
                **error.details,
            )
        if (
            frame_capture_started_at
            and capture_started_at < frame_capture_started_at[-1]
        ):
            stop_timing(
                "authoritative_live_capture_clock_invalid",
                timing_failure="clock_moved_backward_after_interval_wait",
                frame_index=index,
                observed_capture_started_at=capture_started_at,
            )
        if next_capture_not_before is not None and (
            capture_started_at < next_capture_not_before
        ):
            stop_timing(
                "authoritative_live_capture_interval_unverified",
                timing_failure="capture_started_before_scheduled_interval",
                frame_index=index,
                scheduled_capture_not_before=next_capture_not_before,
                observed_capture_started_at=capture_started_at,
            )
        frame_capture_started_at.append(capture_started_at)
        if index == 1 and defer_middle_popup and first_frame_ocr_bundle is not None:
            # Frame zero includes OCR that later captures do not perform. Use
            # that measured cost to decide where to do the middle popup check.
            # This only schedules work; both actual freshness gates still apply.
            first_interval_ms = (capture_started_at - frame_capture_started_at[0]) * 1000.0
            first_ocr_ms = first_frame_ocr_bundle.get("elapsed_ms")
            if (type(first_ocr_ms) in {int, float} and math.isfinite(first_ocr_ms)
                    and 0 <= first_ocr_ms <= first_interval_ms):
                projected_capture_span_ms = 2 * first_interval_ms - first_ocr_ms
                defer_middle_popup = projected_capture_span_ms > observation_window_ms
            if split_middle_popup:
                # A whole popup can fit while forcing the longer OCR read
                # past the final capture (1655). Give OCR the first measured
                # budget, then fill remaining time with measured popup parts.
                defer_middle_popup = True
        next_capture_not_before = (
            capture_started_at + SELECTION_FRAME_INTERVAL_SECONDS
        )
        frame = context.tasker.controller.post_screencap().wait().get()
        frames_buffer.append(frame)
        include_popup = index != 1 or not defer_middle_popup
        if scene_batch is not None:
            health, playable, secondary, popup = (
                scene_batch.observe_with_popup_parts(frame) if index == 0 and split_middle_popup
                else scene_batch.observe(frame, include_popup=include_popup))
            health_votes += int(health)
            playable_votes += int(playable)
            secondary_votes += int(secondary)
            popup_votes += int(popup)
        else:
            health_votes += int(_recognition_hit(context, "ProduceRecognitionHealthFlag", frame))
            playable_votes += int(_recognition_hit(context, "ProduceRecognitionSkipRound", frame))
            secondary_votes += int(_recognition_hit(context, "ProduceRecognitionChooseMoveCards", frame))
            if include_popup:
                popup_votes += int(_recognition_hit(context, "ProduceButton", frame))
        detail = context.run_recognition("ProduceRecognitionCards", frame)
        results = (
            list(getattr(detail, "all_results", ()) or ())
            if detail and detail.hit
            else []
        )
        try:
            observed_candidates = _normalise_observed_live_candidates(context, frame, results)
        except LiveSelectionError as error:
            if (len(error.events) == 1
                    and error.events[0].get('reason') == 'select_label_accent_unresolved'
                    and error.events[0].get('frame_id') == frame_identifier(frame)):
                raise LiveSelectionError(str(error), events=(dict(error.events[0],
                    capture_phase=phase, capture_frame_ids=[frame_identifier(f) for f in frames_buffer],
                    capture_retry_index=retry_index, select_retries_remaining=select_retries_remaining),)) from error
            raise
        frame_candidates.append(observed_candidates)
        if (index == 0 and (expected_hand_count == 3 or bound_continuation)
                and len(observed_candidates) == expected_hand_count
                and getattr(context, 'gkh_hif_opening_state', None) is not None):
            if bound_continuation and expected_hand_count == 3 and expected_selected_slot is not None:
                prework_finished = time.perf_counter_ns() / 1_000_000_000.0
                if math.isfinite(prework_finished) and prework_finished >= capture_started_at:
                    first_ocr_prework_elapsed_ms = (prework_finished-capture_started_at)*1000.0
                    # Reserve the existing 80 ms for each capture interval.
                    # 1044's same-frame title reuse reduces first prework to
                    # ~444 ms, but reinserting ~89 ms of OCR would still put its
                    # recorded span above 1000 ms. Keep that saved headroom.
                    if first_ocr_prework_elapsed_ms >= (
                            observation_window_ms-LIVE_SELECTED_TRIPLE_CAPTURE_RESERVE_MS)/2:
                        defer_middle_popup = False
                        first_ocr_scheduling_reason = 'selected_three_card_first_frame_load'
                        continue
                    first_ocr_scheduling_reason = 'selected_three_card_first_frame_within_budget'
            # Process frame zero early: this saves its OCR work after the
            # final capture while keeping the full evidence span below its gate.
            frame_ids[0] = frame_identifier(frame)
            first_frame_ocr_bundle = _observe_authoritative_live_ocr_bundle(
                context, frame, observed_candidates, frame_id=frame_ids[0],
                absolute_min_confidence=max(float(params['debug_resource_absolute_min_confidence']),
                    LIVE_AUTHORITATIVE_OCR_MOSAIC_ABSOLUTE_MIN_CONFIDENCE),
                preview_delta_min_confidence=float(params['debug_preview_delta_min_confidence']),
                require_fields=False)
            if (bound_continuation and expected_hand_count == 4
                    and expected_selected_slot is not None
                    and first_frame_ocr_bundle.get('accepted') is True):
                cost_started = time.perf_counter_ns()
                first_elapsed_ms = (cost_started/1e9-capture_started_at)*1000
                first_ocr_ms = first_frame_ocr_bundle.get('elapsed_ms')
                # Reserve both remaining common reads and the existing two
                # capture intervals before spending time on optional cost
                # prework. Deferred cost still uses the same local OCR proxy.
                if (type(first_ocr_ms) in (int, float) and math.isfinite(first_ocr_ms)
                        and math.isfinite(first_elapsed_ms)
                        and 0 <= first_ocr_ms <= first_elapsed_ms
                        and first_elapsed_ms+2*first_ocr_ms+LIVE_SELECTED_TRIPLE_CAPTURE_RESERVE_MS
                        <= observation_window_ms):
                    cost = _ocr_selected_effective_cost(cost_context, frame,
                        observed_candidates[expected_selected_slot])
                    first_frame_ocr_bundle['selected_effective_cost'] = dict(
                        slot=expected_selected_slot, observation=cost)
                    first_frame_ocr_bundle['selected_cost_elapsed_ms'] = (
                        time.perf_counter_ns()-cost_started)/1_000_000.0
        elif (index == 1 and bound_continuation
                and expected_hand_count in (2,3,4,5) and len(observed_candidates) == expected_hand_count
                and (expected_selected_slot is None or expected_hand_count != 2 or defer_middle_popup)
                and _same_live_candidate_geometry(frame_candidates[0], observed_candidates)
                and first_frame_ocr_bundle is not None
                and first_frame_ocr_bundle.get('accepted') is True):
            first_ocr_ms = first_frame_ocr_bundle.get('elapsed_ms')
            # Selected pairs can need an extra mid-frame SELECT/shield read.
            # Only spend capture time on their OCR after moving popup work out.
            # Scheduling alone uses this sample; the authoritative capture
            # timestamps and both actual freshness gates remain unchanged.
            scheduling_now = time.perf_counter_ns() / 1_000_000_000.0
            if (type(first_ocr_ms) in {int, float} and math.isfinite(first_ocr_ms)
                    and first_ocr_ms >= 0 and math.isfinite(scheduling_now)
                    and scheduling_now >= capture_started_at):
                middle_ocr_projected_capture_span_ms = (
                    (max(scheduling_now, next_capture_not_before) - frame_capture_started_at[0])
                    * 1000.0 + first_ocr_ms)
                # 1040's in-capture popup path underestimated this span by
                # 69.693 ms. Reserve scheduling room instead of filling the
                # deadline. 1644's selected triple underestimated middle OCR
                # by 35.873 ms even with its popup deferred; keep the same
                # reserve there, moving only OCR work after the last capture.
                middle_ocr_capture_reserve_ms = (
                    0.0 if defer_middle_popup and not (
                        expected_hand_count == 3 and expected_selected_slot is not None)
                    else LIVE_MIDDLE_OCR_INLINE_POPUP_RESERVE_MS)
                if (middle_ocr_projected_capture_span_ms + middle_ocr_capture_reserve_ms
                        <= observation_window_ms):
                    frame_ids[1] = frame_identifier(frame)
                    middle_frame_ocr_bundle = _observe_authoritative_live_ocr_bundle(
                        context, frame, observed_candidates, frame_id=frame_ids[1],
                        absolute_min_confidence=max(float(params['debug_resource_absolute_min_confidence']),
                            LIVE_AUTHORITATIVE_OCR_MOSAIC_ABSOLUTE_MIN_CONFIDENCE),
                        preview_delta_min_confidence=float(params['debug_preview_delta_min_confidence']),
                        require_fields=False)
            if (expected_hand_count == 4 and expected_selected_slot is not None
                    and middle_frame_ocr_bundle is not None
                    and middle_frame_ocr_bundle.get('accepted') is True):
                cost_ms = first_frame_ocr_bundle.get('selected_cost_elapsed_ms')
                elapsed_ms = (time.perf_counter_ns()/1e9-frame_capture_started_at[0])*1000
                if (type(cost_ms) in (int, float) and math.isfinite(cost_ms) and cost_ms >= 0
                        and math.isfinite(elapsed_ms) and elapsed_ms >= 0
                        and elapsed_ms+cost_ms+middle_ocr_capture_reserve_ms <= observation_window_ms):
                    cost = _ocr_selected_effective_cost(cost_context, frame,
                        observed_candidates[expected_selected_slot])
                    middle_frame_ocr_bundle['selected_effective_cost'] = dict(
                        slot=expected_selected_slot, observation=cost)
        if index == 1 and defer_middle_popup and split_middle_popup:
            # Move only a measured same-frame prefix into remaining capture
            # time. The suffix still runs after capture; neither part alone
            # supplies a negative vote or changes either configured deadline.
            prefix_ms = scene_batch.popup_prefix_elapsed_ms
            now = time.perf_counter_ns()/1e9
            if (type(prefix_ms) in (int, float) and math.isfinite(prefix_ms) and prefix_ms >= 0
                    and math.isfinite(now) and now >= capture_started_at):
                middle_popup_projected_capture_span_ms = (
                    max(next_capture_not_before, now+prefix_ms/1000.)-frame_capture_started_at[0])*1000.
                if middle_popup_projected_capture_span_ms <= observation_window_ms:
                    scene_batch.start_popup(frame)
                    middle_popup_started = True
                    suffix_ms = scene_batch.popup_suffix_elapsed_ms
                    now = time.perf_counter_ns()/1e9
                    if (type(suffix_ms) in (int, float) and math.isfinite(suffix_ms) and suffix_ms >= 0
                            and math.isfinite(now) and now >= capture_started_at):
                        middle_popup_suffix_projected_capture_span_ms = (
                            max(next_capture_not_before, now+suffix_ms/1000.)-frame_capture_started_at[0])*1000.
                        if middle_popup_suffix_projected_capture_span_ms <= observation_window_ms:
                            popup_votes += int(scene_batch.finish_popup(frame))
                            middle_popup_completed = True
    frames = tuple(frames_buffer)
    # Defer only when the observed capture cost predicts insufficient room.
    # Every frame is still checked before input and both deadlines are unchanged.
    if defer_middle_popup and not middle_popup_completed:
        popup_votes += int(scene_batch.finish_popup(frames[1]) if middle_popup_started
                           else scene_batch.popup_hit(frames[1]) if scene_batch is not None
                           else _recognition_hit(context, "ProduceButton", frames[1]))
    reconciled_candidates, fragment_reconciliation = (
        _reconcile_single_live_detector_fragment(
            tuple(frame_candidates),
            expected_hand_count=expected_hand_count,
        )
    )
    if expected_selected_slot is None and fragment_reconciliation is None:
        reconciled_candidates, fragment_reconciliation = _reconcile_hif_unselected_detector_gap(
            context, frames, reconciled_candidates)
    if fragment_reconciliation is not None:
        frame_candidates = list(reconciled_candidates)
        # Either early frame may have anomalous detector boxes. Re-read its
        # OCR and SELECT evidence against the reconciled slots below.
        select_observations = [None, None, None]
        select_evidence = [None, None, None]
        first_frame_analysis_error = None
        first_frame_ocr_bundle = None
        middle_frame_ocr_bundle = None
    for index, frame in enumerate(frames):
        if frame_ids[index] is None and not (
            index == 0 and first_frame_analysis_error is not None
        ):
            frame_ids[index] = frame_identifier(frame)
    resolved_frame_ids = [frame_id for frame_id in frame_ids if frame_id is not None]
    captured_at = frame_capture_started_at[-1]
    adjacent_capture_intervals_seconds = [
        current - previous
        for previous, current in zip(
            frame_capture_started_at[:-1],
            frame_capture_started_at[1:],
            strict=True,
        )
    ]
    evidence_window_age_seconds = (
        frame_capture_started_at[-1] - frame_capture_started_at[0]
    )
    input_deadline = captured_at + input_window_ms / 1000.0
    if (
        not all(
            math.isfinite(interval)
            for interval in adjacent_capture_intervals_seconds
        )
        or not math.isfinite(evidence_window_age_seconds)
        or not math.isfinite(input_deadline)
    ):
        stop_timing(
            "authoritative_live_capture_clock_invalid",
            timing_failure="derived_authorization_time_not_finite",
        )
    if (
        any(interval < 0.0 for interval in adjacent_capture_intervals_seconds)
        or evidence_window_age_seconds < 0.0
        or input_deadline < captured_at
    ):
        stop_timing(
            "authoritative_live_capture_clock_invalid",
            timing_failure="derived_authorization_time_moved_backward",
        )
    interval_tolerance_seconds = 1e-9
    if any(
        interval + interval_tolerance_seconds
        < SELECTION_FRAME_INTERVAL_SECONDS
        for interval in adjacent_capture_intervals_seconds
    ) or (
        evidence_window_age_seconds + interval_tolerance_seconds
        < 2 * SELECTION_FRAME_INTERVAL_SECONDS
    ):
        stop_timing(
            "authoritative_live_capture_interval_unverified",
            timing_failure="authoritative_capture_span_below_minimum",
            adjacent_capture_intervals_ms=[
                interval * 1000.0
                for interval in adjacent_capture_intervals_seconds
            ],
            evidence_window_age_ms=evidence_window_age_seconds * 1000.0,
        )
    evidence_window_age_ms = evidence_window_age_seconds * 1000.0
    adjacent_capture_intervals_ms = [
        interval * 1000.0 for interval in adjacent_capture_intervals_seconds
    ]
    if (
        not math.isfinite(evidence_window_age_ms)
        or not all(
            math.isfinite(interval) for interval in adjacent_capture_intervals_ms
        )
    ):
        stop_timing(
            "authoritative_live_capture_clock_invalid",
            timing_failure="authorization_time_milliseconds_not_finite",
        )
    event: dict[str, Any] = {
        "phase": phase,
        "frame_ids": resolved_frame_ids,
        "evidence_window_started_at": frame_capture_started_at[0],
        "evidence_window_age_ms": evidence_window_age_ms,
        "adjacent_capture_intervals_ms": adjacent_capture_intervals_ms,
        "authoritative_capture_started_at": captured_at,
        "expected_selected_slot": expected_selected_slot,
        "allow_selected_score_preview_change": bool(
            phase == "authoritative_refresh_b"
            and allow_selected_score_preview_change
        ),
        "click_dispatched": False,
        "input_may_have_been_dispatched": prior_input_may_have_been_dispatched,
        "retry_index": retry_index,
        "first_ocr_prework_elapsed_ms": first_ocr_prework_elapsed_ms,
        "first_ocr_scheduling_reason": first_ocr_scheduling_reason,
        "ocr_acquisition_schedule": ('first_two_frames_during_capture' if middle_frame_ocr_bundle is not None
                                     else 'first_frame_during_capture' if first_frame_ocr_bundle is not None
                                     else 'after_three_frames'),
        "middle_ocr_projected_capture_span_ms": middle_ocr_projected_capture_span_ms,
        "middle_ocr_capture_reserve_ms": middle_ocr_capture_reserve_ms,
        "middle_popup_schedule": ('during_capture_after_ocr' if middle_popup_completed
            else 'split_during_and_after_capture' if middle_popup_started
            else 'after_three_frames' if defer_middle_popup else 'during_capture'),
        "middle_popup_projected_capture_span_ms": middle_popup_projected_capture_span_ms,
        "middle_popup_suffix_projected_capture_span_ms": middle_popup_suffix_projected_capture_span_ms,
        "projected_capture_span_ms": projected_capture_span_ms,
        "scene_acquisition": scene_batch.events if scene_batch is not None else [],
    }
    if isinstance(cost_context, _SelectedCostOcrContext):
        event['selected_cost_ocr_reuse'] = cost_context.counts
    if fragment_reconciliation is not None:
        event["candidate_fragment_reconciliation"] = fragment_reconciliation

    def stop(reason: str) -> None:
        event.update({"accepted": False, "reason": reason})
        raise LiveSelectionError(reason, events=(event,))

    event.update(
        {
            "health_votes": health_votes,
            "playable_votes": playable_votes,
            "secondary_votes": secondary_votes,
            "popup_votes": popup_votes,
            "candidate_boxes": [
                [list(candidate.box) for candidate in observed]
                for observed in frame_candidates
            ],
        }
    )
    if observation_window_ms != max_snapshot_age_ms:
        event["observation_window_ms"] = observation_window_ms
    if input_window_ms != max_snapshot_age_ms:
        event.update(input_window_ms=input_window_ms, hif_policy_profile=params['hif_policy_profile'])
    if evidence_window_age_ms > observation_window_ms:
        if (bound_continuation and phase=='authoritative_refresh_a'
                and expected_selected_slot is None and expected_generation is None
                and allow_input_free_snapshot_retry is True and snapshot_retries_remaining==1
                and retry_index==0 and isinstance(input_click_counts,tuple)
                and len(input_click_counts)==3
                and all(type(count) is int and count==0 for count in input_click_counts)):
            event.update(accepted=False,reason='authoritative_live_evidence_window_expired',
                         snapshot_retry_triggered=True,snapshot_retry_kind='expired_preinput_full_batch')
            try:
                retry = _capture_verified_live_refresh(context,identity_certificate,
                    expected_selected_slot=None,expected_generation=None,params=params,plan=plan,
                    phase=phase,recognizer=recognizer,catalog=catalog,policy=policy,safety=safety,
                    expected_hand_count=expected_hand_count,resource_baseline=resource_baseline,
                    select_retries_remaining=0,hand_count_retries_remaining=0,
                    allow_input_free_snapshot_retry=False,snapshot_retries_remaining=0,
                    input_click_counts=input_click_counts,retry_index=1)
            except LiveSelectionError as error:
                raise LiveSelectionError(str(error),events=(event,*error.events)) from error
            retry['event'].update(snapshot_retry_count=1,snapshot_retries_remaining=0,
                prior_attempts=[event,*retry['event'].get('prior_attempts',[])])
            return retry
        stop("authoritative_live_evidence_window_expired")
    health_state_stable = health_votes == 3 or (
        expected_selected_slot is not None and health_votes == 0
    )
    if not health_state_stable or playable_votes != 3:
        stop("authoritative_live_anchors_not_unanimous")
    if secondary_votes or popup_votes:
        stop("authoritative_live_secondary_ui_visible")
    observed_hand_counts = [len(observed) for observed in frame_candidates]
    certificate_hand_count = len(identity_certificate)
    if any(count != certificate_hand_count for count in observed_hand_counts):
        matching_hand_count_frames = sum(
            count == certificate_hand_count for count in observed_hand_counts
        )
        mismatched_hand_counts = [
            count
            for count in observed_hand_counts
            if count != certificate_hand_count
        ]
        retryable_transient_hand_count = bool(
            hand_count_retries_remaining > 0
            and phase == "authoritative_refresh_a"
            and expected_selected_slot is None
            and event["click_dispatched"] is False
            and matching_hand_count_frames == 2
            and len(mismatched_hand_counts) == 1
            and (abs(mismatched_hand_counts[0] - certificate_hand_count) == 1
                 or ((continuation_pair or (bound_continuation and certificate_hand_count in (3, 4, 5)))
                     and mismatched_hand_counts[0] == 0
                     and _same_live_candidate_geometry(*[
                         row for row in frame_candidates if len(row) == certificate_hand_count])))
        )
        if (bound_continuation and phase == 'authoritative_refresh_a'
                and expected_selected_slot is None and expected_generation is None
                and hand_count_retries_remaining > 0 and retry_index == 0
                and allow_input_free_snapshot_retry is True and snapshot_retries_remaining == 1
                and isinstance(input_click_counts, tuple) and len(input_click_counts) == 3
                and all(type(count) is int and count == 0 for count in input_click_counts)
                and matching_hand_count_frames == 1 and mismatched_hand_counts == [0, 0]):
            # Empty detector rows are unresolved, not two confirmed empty
            # hands. One complete row justifies the same single whole-set
            # retry; it never fills either missing frame or authorizes input.
            retryable_transient_hand_count = True
        if (hand_count_retries_remaining > 0 and phase == "authoritative_refresh_a"
                and bound_continuation and expected_selected_slot is not None
                and allow_input_free_snapshot_retry is True
                and isinstance(input_click_counts, tuple) and len(input_click_counts) == 3
                and all(type(count) is int and count == 0 for count in input_click_counts)
                and (_hif_selected_five_frame_missing(
                    event["candidate_boxes"], expected_selected_slot, certificate_hand_count)
                     or _hif_selected_unresolved_layout_batch(
                         event["candidate_boxes"], expected_selected_slot, certificate_hand_count))):
            # A recovered selection can contain unresolved detector fragments.
            # Discard this whole batch; the existing one-retry path below
            # must rebuild every input check from three new frames.
            retryable_transient_hand_count = True
        if retryable_transient_hand_count:
            event.update(
                {
                    "accepted": False,
                    "reason": "authoritative_live_hand_count_transient",
                }
            )
            time.sleep(SELECTION_SETTLE_SECONDS)
            try:
                retry = _capture_verified_live_refresh(
                    context,
                    identity_certificate,
                    expected_selected_slot=expected_selected_slot,
                    expected_generation=expected_generation,
                    params=params,
                    plan=plan,
                    phase=phase,
                    recognizer=recognizer,
                    catalog=catalog,
                    policy=policy,
                    safety=safety,
                    expected_hand_count=expected_hand_count,
                    resource_baseline=resource_baseline,
                    selected_transition_verified=selected_transition_verified,
                    selected_identity_journal=selected_identity_journal,
                    allow_selected_score_preview_change=(
                        allow_selected_score_preview_change
                    ),
                    select_retries_remaining=select_retries_remaining,
                    hand_count_retries_remaining=(
                        hand_count_retries_remaining - 1
                    ),
                    allow_input_free_snapshot_retry=False,
                    snapshot_retries_remaining=0,
                    input_click_counts=None,
                    retry_index=retry_index + 1,
                )
            except LiveSelectionError as error:
                raise LiveSelectionError(
                    str(error),
                    events=(event, *error.events),
                ) from error
            retry["event"]["prior_attempts"] = [
                event,
                *list(retry["event"].get("prior_attempts", [])),
            ]
            return retry
        stop("authoritative_live_hand_count_changed")
    if not frame_candidates or not all(
        _same_live_candidate_geometry(frame_candidates[0], observed)
        for observed in frame_candidates[1:]
    ):
        stop("authoritative_live_hand_geometry_changed")
    event["geometry_stable"] = True
    if first_frame_analysis_error is not None:
        raise first_frame_analysis_error

    authoritative_ocr_evidence: tuple[dict[str, Any], ...] | None = None
    if expected_hand_count == 3 or bound_continuation:
        authoritative_mosaic_required = bool(
            phase in {"authoritative_refresh_a", "authoritative_refresh_b"}
            and type(params.get("debug_expected_initial_hand_count")) is int
            and params.get("debug_expected_initial_hand_count") == expected_hand_count
        )
        absolute_threshold = max(
            float(params["debug_resource_absolute_min_confidence"]),
            LIVE_AUTHORITATIVE_OCR_MOSAIC_ABSOLUTE_MIN_CONFIDENCE,
        )
        preview_delta_threshold = float(
            params["debug_preview_delta_min_confidence"]
        )
        acquired_bundles: list[dict[str, Any]] = []
        for index, (frame, observed, frame_id) in enumerate(
            zip(frames, frame_candidates, resolved_frame_ids, strict=True)
        ):
            cached_bundle = first_frame_ocr_bundle if index == 0 else middle_frame_ocr_bundle if index == 1 else None
            bundle = cached_bundle if cached_bundle is not None else _observe_authoritative_live_ocr_bundle(
                context, frame, observed, frame_id=frame_id,
                absolute_min_confidence=absolute_threshold,
                preview_delta_min_confidence=preview_delta_threshold,
                require_fields=index == 2)
            if bundle.get("supported") is not True:
                acquired_bundles = []
                event["ocr_source_fallback"] = {
                    "frame_index": index,
                    "reason": bundle.get("reason"),
                }
                if authoritative_mosaic_required:
                    event["ocr_acquisition_mode"] = (
                        "authoritative_mosaic_source_unavailable"
                    )
                    event["ocr_source_fallback_stop"] = dict(
                        event["ocr_source_fallback"]
                    )
                    stop("authoritative_live_ocr_mosaic_source_unavailable")
                event["ocr_acquisition_mode"] = "legacy_direct_source_fallback"
                break
            if bundle.get("accepted") is not True:
                event["ocr_acquisition_mode"] = "authoritative_mosaic"
                event["ocr_contradiction_stop"] = {
                    "frame_index": index,
                    "reason": bundle.get("reason"),
                    "common": bundle.get("common"),
                    "delta_common": bundle.get("delta_common"),
                    "fallback_conflict": bundle.get("fallback_conflict"),
                }
                stop("authoritative_live_ocr_mosaic_contradiction")
            if not _authoritative_live_ocr_bundle_bound(
                bundle,
                frame_id=frame_id,
                candidates=observed,
            ):
                event["ocr_acquisition_mode"] = "authoritative_mosaic"
                event["ocr_contradiction_stop"] = {
                    "frame_index": index,
                    "reason": "authoritative_live_ocr_bundle_binding_mismatch",
                }
                stop("authoritative_live_ocr_mosaic_contradiction")
            acquired_bundles.append(bundle)
        if len(acquired_bundles) == 3:
            authoritative_ocr_evidence = tuple(acquired_bundles)
            event["ocr_acquisition_mode"] = "authoritative_three_frame_mosaic"
            event["ocr_fallbacks"] = [
                {
                    "frame_index": index,
                    "fallbacks": list(bundle.get("fallbacks", [])),
                }
                for index, bundle in enumerate(authoritative_ocr_evidence)
                if bundle.get("fallbacks")
            ]
            event["ocr_acquisition_timing_ms"] = [
                bundle.get("elapsed_ms") for bundle in authoritative_ocr_evidence
            ]

    for index, (frame, observed) in enumerate(
        zip(frames, frame_candidates, strict=True)
    ):
        if select_observations[index] is None or select_evidence[index] is None:
            if authoritative_ocr_evidence is not None:
                slots, evidence = _observe_expected_selected_live_slots(
                    context,
                    frame,
                    observed,
                    expected_selected_slot,
                    ocr_slots=tuple(
                        authoritative_ocr_evidence[index]["select_slots"]
                    ),
                )
            else:
                slots, evidence = _observe_expected_selected_live_slots(
                    context,
                    frame,
                    observed,
                    expected_selected_slot,
                )
            if authoritative_ocr_evidence is not None:
                evidence["ocr_acquisition_mode"] = authoritative_ocr_evidence[index].get(
                    "select_acquisition_mode", "common_mosaic")
            select_observations[index] = slots
            select_evidence[index] = evidence
    resolved_select_observations = [
        observation
        for observation in select_observations
        if observation is not None
    ]
    resolved_select_evidence = [
        evidence for evidence in select_evidence if evidence is not None
    ]
    if (
        len(resolved_frame_ids) != 3
        or len(resolved_select_observations) != 3
        or len(resolved_select_evidence) != 3
    ):
        stop("authoritative_live_frame_evidence_incomplete")
    event["frame_ids"] = resolved_frame_ids
    select_consensus = resolve_live_selected_slot_consensus(
        resolved_select_observations
    )
    event["select_consensus"] = select_consensus.to_dict()
    event["select_evidence"] = resolved_select_evidence
    if (
        not select_consensus.accepted
        or select_consensus.selected_slot != expected_selected_slot
    ):
        observed_selected_slots = {
            slot
            for observation in select_consensus.observations
            for slot in observation
        }
        retryable_missing_select = bool(
            select_retries_remaining > 0
            and expected_selected_slot is not None
            and select_consensus.reason == "select_frames_disagree"
            and observed_selected_slots == {expected_selected_slot}
        )
        if retryable_missing_select:
            event.update(
                {
                    "accepted": False,
                    "reason": "authoritative_live_select_state_transient",
                }
            )
            time.sleep(SELECTION_SETTLE_SECONDS)
            try:
                retry = _capture_verified_live_refresh(
                    context,
                    identity_certificate,
                    expected_selected_slot=expected_selected_slot,
                    expected_generation=expected_generation,
                    params=params,
                    plan=plan,
                    phase=phase,
                    recognizer=recognizer,
                    catalog=catalog,
                    policy=policy,
                    safety=safety,
                    expected_hand_count=expected_hand_count,
                    resource_baseline=resource_baseline,
                    selected_transition_verified=selected_transition_verified,
                    selected_identity_journal=selected_identity_journal,
                    allow_selected_score_preview_change=(
                        allow_selected_score_preview_change
                    ),
                    select_retries_remaining=select_retries_remaining - 1,
                    hand_count_retries_remaining=hand_count_retries_remaining,
                    allow_input_free_snapshot_retry=False,
                    snapshot_retries_remaining=0,
                    input_click_counts=None,
                    retry_index=retry_index + 1,
                )
            except LiveSelectionError as error:
                raise LiveSelectionError(
                    str(error),
                    events=(event, *error.events),
                ) from error
            retry["event"]["prior_attempts"] = [
                event,
                *list(retry["event"].get("prior_attempts", [])),
            ]
            return retry
        stop("authoritative_live_select_state_changed")

    unselected_geometry_authorized = bool(
        expected_selected_slot is None
        and len(resolved_select_evidence) == 3
        and all(
            evidence.get("geometry_available") is True
            and evidence.get("geometry_ambiguous") is False
            and evidence.get("geometry_raised_slots") == []
            and evidence.get("resolved_slots") == []
            for evidence in resolved_select_evidence
        )
    )
    event["unselected_geometry_authorized"] = unselected_geometry_authorized
    if expected_selected_slot is None and not unselected_geometry_authorized:
        stop("authoritative_live_unselected_geometry_unverified")

    authoritative_image = frames[-1]
    authoritative_candidates = frame_candidates[-1]
    authoritative_frame_id = resolved_frame_ids[-1]
    try:
        fresh_predictions = classify_live_card_views(
            recognizer,
            authoritative_image,
            authoritative_candidates,
            frame_id=authoritative_frame_id,
            plan=plan,
            acceptance_threshold=float(params["debug_identity_threshold"]),
            min_margin=float(params["debug_identity_min_margin"]),
            art_preprocessing_evidence=event.setdefault('art_preprocessing_evidence', []),
            **_live_card_art_profile(context, authoritative_candidates, expected_selected_slot),
        )
    except LiveCardLayoutError as error:
        event['card_layout_error'] = str(error)
        if error.__cause__ is not None:
            event['card_layout_error_cause'] = str(error.__cause__)
        stop("authoritative_live_card_layout_invalid")
    raw_fresh_predictions = fresh_predictions
    fresh_predictions = _corroborate_bound_hif_families(context, authoritative_image,
        authoritative_candidates, fresh_predictions, selected_slot=expected_selected_slot,
        minimum_margin=float(params['debug_identity_min_margin']), evidence=event,
        selected_neighbor_frames=frames, selected_neighbor_candidates=frame_candidates,
        minimum_confidence=params.get('minimum_identity_confidence'))
    for proof in event.get('selection_common_art_families', ()):
        matched_at = max(frame_capture_started_at[resolved_frame_ids.index(fid)]
                         for fid in proof['matching_frame_ids'])
        captured_at = min(captured_at, matched_at)
        input_deadline = min(input_deadline, matched_at+input_window_ms/1000.)
    if event.get('selection_common_art_families'):
        event['identity_evidence_started_at'] = captured_at
    fresh_predictions = _corroborate_bound_hif_selected_family(context, frames,
        fresh_predictions, identity_certificate, selected_slot=expected_selected_slot,
        minimum_margin=float(params['debug_identity_min_margin']), evidence=event,
        catalog=catalog, plan=plan, candidates=authoritative_candidates,
        minimum_confidence=params.get('minimum_identity_confidence'))
    if len(fresh_predictions) != len(identity_certificate):
        stop("authoritative_live_identity_count_changed")

    certificate_by_slot = {
        int(certificate["slot"]): certificate
        for certificate in identity_certificate
    }
    fresh_by_slot = {
        prediction.slot: prediction for prediction in fresh_predictions
    }
    event["identity_revalidation"] = []
    revalidation_by_slot: dict[int, dict[str, Any]] = {}
    failed_slots: list[int] = []
    from live_decision.hif_bound_title import family as identity_family
    from live_decision.hif_bound_title import event_proof

    for candidate in authoritative_candidates:
        certificate = certificate_by_slot.get(candidate.slot)
        prediction = fresh_by_slot.get(candidate.slot)
        family = (identity_family(prediction, event_proof(event, prediction))
                  if prediction is not None else ())
        revalidation = {
            "slot": candidate.slot,
            "expected_family": (
                list(certificate["family"]) if certificate is not None else None
            ),
            "fresh_family": list(family),
            "fresh_prediction": prediction.to_dict() if prediction is not None else None,
        }
        standard_accepted = not (
            certificate is None
            or prediction is None
            or not prediction.accepted
            or family != tuple(certificate["family"])
        )
        revalidation.update(
            {
                "standard_accepted": standard_accepted,
                "carried": False,
                "accepted": standard_accepted,
                "acceptance_mode": (
                    "fresh_exact_family" if standard_accepted else None
                ),
                "carry_rejection_reasons": [],
            }
        )
        event["identity_revalidation"].append(revalidation)
        revalidation_by_slot[candidate.slot] = revalidation
        if not standard_accepted:
            failed_slots.append(candidate.slot)
    neighbor_continuity = _hif_selected_neighbor_continuity(context,
        frames=frames, rows=frame_candidates, predictions=raw_fresh_predictions,
        certificates=certificate_by_slot, phase=phase,
        selected_slot=expected_selected_slot, expected_generation=expected_generation,
        selected_transition_verified=selected_transition_verified)
    if neighbor_continuity is not None:
        slot = neighbor_continuity['slot']
        if slot in failed_slots:
            revalidation_by_slot[slot].update(accepted=True, carried=True,
                acceptance_mode='same_transaction_neighbor_exact_art',
                fresh_family=list(neighbor_continuity['family']),
                neighbor_continuity=neighbor_continuity)
            failed_slots.remove(slot)
    selected_recovery = getattr(context, 'gkh_hif_gray_selected_source', None)
    if (selected_recovery is None and phase == 'authoritative_refresh_b'
            and selected_transition_verified is True
            and isinstance(selected_identity_journal, dict)
            and selected_identity_journal.get('refresh_a', {}).get('generation') == expected_generation):
        # A normal, still-unsubmitted SELECT has the same known hand as a
        # recovered SELECT. Reuse its actual A and dispatch, not a new visual
        # verdict for an occluded non-target card.
        selected_recovery = selected_identity_journal
    if expected_selected_slot is not None and isinstance(selected_recovery, dict):
        for slot in tuple(failed_slots):
            if slot != expected_selected_slot + 1:
                continue
            try:
                carried, proof = _carry_hif_selected_neighbor(context, fresh_by_slot[slot],
                    journal=selected_recovery,
                    evidence=dict(event, geometry_stable=True,
                        select_consensus=select_consensus.to_dict(),
                        select_evidence=resolved_select_evidence),
                    identity_threshold=float(params['debug_identity_threshold']),
                    identity_min_margin=float(params['debug_identity_min_margin']))
            except ValueError as error:
                revalidation_by_slot[slot]['carry_rejection_reasons'] = [str(error)]
                continue
            fresh_by_slot[slot] = carried
            revalidation_by_slot[slot].update(proof)
            failed_slots.remove(slot)
    for slot in failed_slots:
        if slot != expected_selected_slot:
            revalidation_by_slot[slot]["carry_rejection_reasons"] = [
                "not_expected_selected_slot"
            ]
    if any(slot != expected_selected_slot for slot in failed_slots):
        from live_decision.hif_bound_title import in_candidate_band, retryable_low_confidence_family

        identity_ambiguity_retry = (bound_continuation and phase == 'authoritative_refresh_a'
            and expected_selected_slot is None and expected_generation is None
            and allow_input_free_snapshot_retry is True and snapshot_retries_remaining == 1
            and retry_index == 0 and isinstance(input_click_counts, tuple)
            and len(input_click_counts) == 3
            and all(type(count) is int and count == 0 for count in input_click_counts)
            and all(in_candidate_band(fresh_by_slot[slot], certificate_by_slot[slot]['family'],
                float(params['debug_identity_min_margin']))
                or retryable_low_confidence_family(fresh_by_slot[slot], certificate_by_slot[slot]['family'],
                    float(params['debug_identity_min_margin'])) for slot in failed_slots))
        if identity_ambiguity_retry:
            # Animation can reorder candidates or lower the same family's confidence. Reject
            # this set and spend the existing single zero-input retry; every
            # family, resource and timestamp must pass again in the new set.
            event.update(accepted=False, reason='authoritative_live_identity_family_changed',
                snapshot_retry_triggered=True, snapshot_retry_kind='weak_preinput_identity_full_batch')
            try:
                retry = _capture_verified_live_refresh(context, identity_certificate,
                    expected_selected_slot=None, expected_generation=None, params=params, plan=plan,
                    phase=phase, recognizer=recognizer, catalog=catalog, policy=policy, safety=safety,
                    expected_hand_count=expected_hand_count, resource_baseline=resource_baseline,
                    select_retries_remaining=0, hand_count_retries_remaining=0,
                    allow_input_free_snapshot_retry=False, snapshot_retries_remaining=0,
                    input_click_counts=input_click_counts, retry_index=1)
            except LiveSelectionError as error:
                raise LiveSelectionError(str(error), events=(event, *error.events)) from error
            retry['event'].update(snapshot_retry_count=1, snapshot_retries_remaining=0,
                prior_attempts=[event, *retry['event'].get('prior_attempts', [])])
            return retry
        stop("authoritative_live_identity_family_changed")

    refresh_upgrade_evidence: dict[int, dict[str, Any]] = {}
    if expected_selected_slot is not None:
        selected_certificate = certificate_by_slot[expected_selected_slot]
        eligible_ids = _eligible_family_ids(
            catalog,
            tuple(selected_certificate["family"]),
            plan,
        )
        carry_requires_title = expected_selected_slot in failed_slots
        if len(eligible_ids) > 1 or carry_requires_title:
            selected_view = build_live_title_views(
                tuple(authoritative_image.shape),
                authoritative_candidates,
                selected_slot=expected_selected_slot,
                hif_layout=isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext),
            )[expected_selected_slot]
            marker_upgrade = resolve_live_title_consensus(
                authoritative_image,
                selected_view,
                frame_id=authoritative_frame_id,
            )
            selected_upgrade = marker_upgrade
            selected_detail = None
            if isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext):
                # A complete HIF popup title distinguishes suffixes from a
                # final glyph such as 力, which also fits the expanded cross.
                selected_upgrade, selected_detail = _hif_selected_upgrade_consensus(
                    context, frames, tuple(selected_certificate["family"]), catalog, plan)
                event["refresh_upgrade_marker_evidence"] = {
                    expected_selected_slot: marker_upgrade.to_dict()}
                event["refresh_upgrade_detail_evidence"] = {
                    expected_selected_slot: selected_detail}
            refresh_upgrade_evidence[expected_selected_slot] = selected_upgrade.to_dict()
            event["refresh_upgrade_evidence"] = refresh_upgrade_evidence
            support_upgrade = None
            if (selected_upgrade.accepted and selected_upgrade.upgraded is True
                    and selected_certificate['upgraded'] is False
                    and marker_upgrade.accepted and marker_upgrade.upgraded is True
                    and len(marker_upgrade.observations) == marker_upgrade.plus_votes == 1
                    and getattr(policy, 'selected_skill_support', None) is not None):
                from live_decision.hif_skill_support import selected_support_visible_upgrade

                support_upgrade = selected_support_visible_upgrade(
                    getattr(context, 'gkh_hif_opening_state', None), policy, selected_certificate,
                    slot=expected_selected_slot, frame_ids=resolved_frame_ids,
                    upgrade_evidence=marker_upgrade.to_dict(),
                    identity_evidence=revalidation_by_slot[expected_selected_slot])
                event['selected_support_visible_upgrade'] = dict(
                    accepted=support_upgrade is not None, proof=support_upgrade)
            if (support_upgrade is None and selected_upgrade.accepted
                    and selected_upgrade.upgraded is True and selected_certificate['upgraded'] is False
                    and getattr(policy, 'selected_single_plus_estimate', None) is not None):
                from live_decision.hif_single_plus import selected_ordinary_plus_visible_upgrade

                support_upgrade = selected_ordinary_plus_visible_upgrade(
                    getattr(context, 'gkh_hif_opening_state', None), policy, selected_certificate,
                    slot=expected_selected_slot, frame_ids=resolved_frame_ids, detail=selected_detail,
                    selection_evidence=dict(frame_ids=resolved_frame_ids, geometry_stable=True,
                        select_consensus=event['select_consensus'], select_evidence=resolved_select_evidence),
                    identity_evidence=revalidation_by_slot[expected_selected_slot])
                event['selected_ordinary_plus_visible_upgrade'] = dict(
                    accepted=support_upgrade is not None, proof=support_upgrade)
            if (
                not selected_upgrade.accepted
                or selected_upgrade.upgraded is not selected_certificate["upgraded"]
                and support_upgrade is None
            ):
                stop("authoritative_live_selected_upgrade_changed")
            if carry_requires_title:
                expected_upgrade_observations = 3 if selected_detail is not None else 1
                unanimous_upgrade = bool(
                    (len(selected_detail["observations"]) if selected_detail is not None
                     else len(selected_upgrade.observations)) == expected_upgrade_observations
                    and (
                        selected_upgrade.plus_votes == expected_upgrade_observations
                        if selected_upgrade.upgraded is True
                        else selected_upgrade.normal_votes == expected_upgrade_observations
                    )
                )
                if not unanimous_upgrade:
                    stop("authoritative_live_selected_upgrade_changed")
            if (carry_requires_title and selected_detail is not None
                    and selected_certificate.get('inspected_art_evidence') is not None):
                # This group already supplied the full detail title. An
                # inspected A source needs that fresh B evidence, not another
                # OCR call or the old four-card image-carry exception.
                corroborated = _corroborate_bound_hif_selected_family(context, frames,
                    tuple(fresh_by_slot[i] for i in range(len(fresh_by_slot))), identity_certificate,
                    selected_slot=expected_selected_slot,
                    minimum_margin=float(params['debug_identity_min_margin']),
                    evidence=event, catalog=catalog, plan=plan, selected_detail=selected_detail,
                    candidates=authoritative_candidates)
                fresh = corroborated[expected_selected_slot]
                if fresh.accepted and identity_family(fresh, event_proof(event, fresh)) == tuple(selected_certificate['family']):
                    fresh_by_slot[expected_selected_slot] = fresh
                    revalidation_by_slot[expected_selected_slot].update(
                        accepted=True, standard_accepted=True, fresh_prediction=fresh.to_dict(),
                        acceptance_mode='fresh_complete_selected_title', carry_rejection_reasons=[])
                    failed_slots.remove(expected_selected_slot)

    if failed_slots:
        if expected_selected_slot is None or failed_slots != [expected_selected_slot]:
            stop("authoritative_live_identity_family_changed")
        selected_certificate = certificate_by_slot.get(expected_selected_slot)
        selected_prediction = fresh_by_slot.get(expected_selected_slot)
        selected_candidate = next(
            (
                candidate
                for candidate in authoritative_candidates
                if candidate.slot == expected_selected_slot
            ),
            None,
        )
        selected_revalidation = revalidation_by_slot[expected_selected_slot]
        rejection_reasons: list[str] = []
        if (
            len(authoritative_candidates) != 4
            or tuple(candidate.slot for candidate in authoritative_candidates)
            != tuple(range(4))
        ):
            rejection_reasons.append("hand_not_exact_four")
        geometry_authoritative = bool(
            len(select_evidence) == 3
            and all(
                evidence.get("geometry_available") is True
                and evidence.get("geometry_ambiguous") is False
                and evidence.get("geometry_slot") == expected_selected_slot
                and evidence.get("geometry_raised_slots")
                == [expected_selected_slot]
                and evidence.get("resolved_slots") == [expected_selected_slot]
                for evidence in select_evidence
            )
        )
        if not geometry_authoritative:
            rejection_reasons.append("selected_geometry_not_authoritative")
        source_prediction = (
            selected_certificate.get("source_prediction")
            if selected_certificate is not None
            else None
        )
        selected_revalidation["carry_source"] = {
            "source_prediction": (
                source_prediction.to_dict() if source_prediction is not None else None
            ),
            "source_selected_slot": (
                selected_certificate.get("source_selected_slot")
                if selected_certificate is not None
                else None
            ),
            "unselected_view_verified": bool(
                selected_certificate is not None
                and selected_certificate.get("unselected_view_verified") is True
            ),
            "certificate_selected_transition_verified": bool(
                selected_certificate is not None
                and selected_certificate.get("selected_transition_verified") is True
            ),
            "refresh_selected_transition_verified": selected_transition_verified,
        }
        if source_prediction is None:
            rejection_reasons.append("source_prediction_missing")
        else:
            source_family = (
                tuple(source_prediction.top_k_card_ids[0])
                if source_prediction.top_k_card_ids
                else ()
            )
            if (
                source_prediction.slot != expected_selected_slot
                or source_prediction.accepted is not True
                or source_prediction.card_id != selected_certificate["card_id"]
                or source_family != tuple(selected_certificate["family"])
            ):
                rejection_reasons.append("source_prediction_invalid")
            source_confidence = source_prediction.confidence
            source_margin = source_prediction.margin
            if (
                isinstance(source_confidence, bool)
                or not isinstance(source_confidence, (int, float))
                or not math.isfinite(float(source_confidence))
                or float(source_confidence)
                < float(params["debug_identity_threshold"])
                or isinstance(source_margin, bool)
                or not isinstance(source_margin, (int, float))
                or not math.isfinite(float(source_margin))
                or float(source_margin)
                < float(params["debug_identity_min_margin"])
            ):
                rejection_reasons.append("source_identity_below_threshold")
        if (
            selected_certificate is None
            or selected_certificate.get("unselected_view_verified") is not True
            or selected_certificate.get("source_selected_slot")
            == expected_selected_slot
        ):
            rejection_reasons.append("source_unselected_view_unverified")
        if not (
            selected_transition_verified
            or (
                selected_certificate is not None
                and selected_certificate.get("selected_transition_verified") is True
            )
        ):
            rejection_reasons.append("selected_transition_unverified")
        if selected_prediction is None:
            rejection_reasons.append("fresh_prediction_missing")
        elif source_prediction is not None:
            hash_names = ("model_sha256", "classes_sha256", "gallery_sha256")
            if any(
                type(getattr(source_prediction, name, None)) is not str
                or not getattr(source_prediction, name)
                or getattr(source_prediction, name)
                != getattr(selected_prediction, name, None)
                for name in hash_names
            ):
                rejection_reasons.append("recognizer_hash_mismatch")
        selected_revalidation["carry_rejection_reasons"] = rejection_reasons
        if rejection_reasons:
            stop("authoritative_live_identity_family_changed")
        carried_prediction = replace(
            source_prediction,
            box=selected_candidate.box,
            detector_label=selected_candidate.detector_label,
            detector_score=selected_candidate.detector_score,
            visible=selected_candidate.visible,
            clickable=selected_candidate.clickable,
            accepted=True,
            reason="live_selected_identity_certificate_carried",
        )
        fresh_by_slot[expected_selected_slot] = carried_prediction
        selected_revalidation.update(
            {
                "carried": True,
                "accepted": True,
                "acceptance_mode": "selected_unselected_certificate",
                "carried_prediction": carried_prediction.to_dict(),
            }
        )

    event["selected_identity_authorized"] = bool(
        expected_selected_slot is not None
        and revalidation_by_slot.get(expected_selected_slot, {}).get("accepted") is True
    )

    fields = (
        _observe_live_fields(
            context,
            authoritative_image,
            authoritative_evidence=authoritative_ocr_evidence[-1],
        )
        if authoritative_ocr_evidence is not None
        else _observe_live_fields(context, authoritative_image)
    )
    if not isinstance(resource_baseline, dict):
        stop("authoritative_live_resource_baseline_missing")
    if expected_selected_slot is None:
        transaction_kwargs = {
            "frame_ids": tuple(resolved_frame_ids),
            "baseline": resource_baseline,
        }
        if authoritative_ocr_evidence is not None:
            transaction_kwargs["authoritative_evidence"] = (
                authoritative_ocr_evidence
            )
        resource_transaction = _observe_unselected_resource_transaction(
            context,
            frames,
            **transaction_kwargs,
        )
    else:
        transaction_kwargs = {
            "frame_ids": tuple(resolved_frame_ids),
            "selected_slot": expected_selected_slot,
            "baseline": resource_baseline,
        }
        if authoritative_ocr_evidence is not None:
            transaction_kwargs["authoritative_evidence"] = (
                authoritative_ocr_evidence
            )
        if callable(getattr(policy,'selected_resource_preview',None)):
            preview_contract = policy.selected_resource_preview(expected_selected_slot)
            if preview_contract is not None:
                opening = getattr(context, 'gkh_hif_opening_state', None)
                selected_identity = certificate_by_slot.get(expected_selected_slot, {})
                preview_identity_matches = preview_contract.get('card_id') == selected_identity.get('card_id')
                if not preview_identity_matches:
                    from live_decision.hif_gray_selection import selected_preview_matches_temporary_identity

                    preview_identity_matches = selected_preview_matches_temporary_identity(
                        opening, policy, preview_contract, selected_identity, expected_selected_slot)
                if (opening is None
                        or preview_contract.get('run_id') != opening.run_id
                        or preview_contract.get('revision') != opening.revision
                        or not preview_identity_matches):
                    stop('authoritative_live_selected_preview_binding_invalid')
                transaction_kwargs['card_effect_preview'] = preview_contract
        resource_transaction = _observe_selected_resource_transaction(
            context,
            frames,
            tuple(frame_candidates),
            cost_context=cost_context,
            **transaction_kwargs,
        )
        selected_certificate = certificate_by_slot.get(expected_selected_slot)
        selected_entry = (
            catalog.entries.get(str(selected_certificate.get("card_id")))
            if isinstance(selected_certificate, dict)
            else None
        )
        selected_features = (
            catalog.features(
                selected_entry.card_id,
                turns_remaining=int(fields["turns_remaining"].get("value") or 0),
            )
            if selected_entry is not None
            else {}
        )
        resource_transaction.update(
            {
                "card_id": selected_entry.card_id if selected_entry is not None else None,
                "catalog_base_cost": (
                    selected_entry.stamina_cost if selected_entry is not None else None
                ),
                "effective_minus_catalog": (
                    resource_transaction["displayed_effective_cost"]
                    - selected_entry.stamina_cost
                    if resource_transaction.get("accepted") is True
                    and selected_entry is not None
                    else None
                ),
                "catalog_stamina_recovery": selected_features.get(
                    "stamina_recovery"
                ),
            }
        )
        if (
            selected_entry is None
            or (selected_features.get("stamina_recovery") not in {0, 0.0}
                and resource_transaction.get('card_effect_preview') is None)
        ):
            event["resource_transaction"] = resource_transaction
            stop("authoritative_live_selected_resource_effect_unsupported")
    if resource_transaction.get("accepted") is not True:
        event["resource_transaction"] = resource_transaction
        from live_decision.opening_observer import (
            isolated_preview_pixel_retry,
            selected_refresh_noise_reobserve_allowed,
        )

        if (getattr(context, 'gkh_hif_opening_state', None) is not None
                and phase == 'authoritative_refresh_b' and type(expected_selected_slot) is int
                and selected_transition_verified and retry_index == 0
                and isolated_preview_pixel_retry(resource_transaction, resource_baseline,
                    params.get('debug_resource_absolute_min_confidence'),
                    selected_slot=expected_selected_slot)):
            event.update(snapshot_retry_triggered=True, snapshot_retry_kind='selected_isolated_pixel_full_batch')
            try:
                retry = _capture_verified_live_refresh(context, identity_certificate,
                    expected_selected_slot=expected_selected_slot, expected_generation=expected_generation,
                    params=params, plan=plan, phase=phase, recognizer=recognizer, catalog=catalog,
                    policy=policy, safety=safety, expected_hand_count=expected_hand_count,
                    resource_baseline=resource_baseline, selected_transition_verified=True,
                    selected_identity_journal=selected_identity_journal,
                    allow_selected_score_preview_change=allow_selected_score_preview_change,
                    select_retries_remaining=0, hand_count_retries_remaining=0, retry_index=1)
            except LiveSelectionError as error:
                raise LiveSelectionError(str(error), events=(event, *error.events)) from error
            retry['event'].update(snapshot_retry_count=1, snapshot_retries_remaining=0,
                prior_attempts=[event, *retry['event'].get('prior_attempts', [])])
            return retry

        recovered_selected_noise = (phase == 'authoritative_refresh_a'
            and type(expected_selected_slot) is int
            and isinstance(getattr(context, 'gkh_hif_gray_selected_source', None), dict)
            and selected_refresh_noise_reobserve_allowed(dict(event,
                accepted=False, reason='authoritative_live_resource_transaction_unresolved'),
                params, expected_selected_slot, expected_hand_count, phase=phase,
                baseline=resource_baseline))
        if (getattr(context, 'gkh_hif_opening_state', None) is not None
                and phase == 'authoritative_refresh_a'
                and (expected_selected_slot is None or recovered_selected_noise)
                and expected_generation is None and allow_input_free_snapshot_retry is True
                and snapshot_retries_remaining == 1 and retry_index == 0
                and isinstance(input_click_counts,tuple) and len(input_click_counts)==3
                and all(type(count) is int and count==0 for count in input_click_counts)
                and (recovered_selected_noise or isolated_preview_pixel_retry(resource_transaction,resource_baseline,
                    params.get('debug_resource_absolute_min_confidence')))):
            event.update(snapshot_retry_triggered=True,snapshot_retry_kind='isolated_preview_pixel_full_batch')
            try:
                retry = _capture_verified_live_refresh(context,identity_certificate,
                    expected_selected_slot=expected_selected_slot,expected_generation=None,params=params,plan=plan,
                    phase=phase,recognizer=recognizer,catalog=catalog,policy=policy,safety=safety,
                    expected_hand_count=expected_hand_count,resource_baseline=resource_baseline,
                    select_retries_remaining=0,hand_count_retries_remaining=0,
                    allow_input_free_snapshot_retry=False,snapshot_retries_remaining=0,
                    input_click_counts=input_click_counts,retry_index=1)
            except LiveSelectionError as error:
                raise LiveSelectionError(str(error),events=(event,*error.events)) from error
            retry['event'].update(snapshot_retry_count=1,snapshot_retries_remaining=0,
                prior_attempts=[event,*retry['event'].get('prior_attempts',[])])
            return retry
        stop("authoritative_live_resource_transaction_unresolved")
    input_click_counts_are_strict_zero = bool(
        isinstance(input_click_counts, tuple)
        and len(input_click_counts) == 3
        and all(type(count) is int and count == 0 for count in input_click_counts)
    )
    confidence_gates = [
        (
            "debug_resource_absolute_min_confidence",
            "resource_absolute_confidence",
            "authoritative_live_resource_absolute_confidence_unapproved",
        ),
        (
            "debug_preview_delta_min_confidence",
            "preview_delta_confidence",
            "authoritative_live_preview_delta_confidence_unapproved",
        ),
    ]
    # A bound model preview already verifies the printed cost against the
    # current rules and both observed resources. Its OCR score is auxiliary;
    # observed-payment-only transactions still need the independent cost gate.
    if (expected_selected_slot is not None
            and resource_transaction.get('card_effect_preview') is None):
        confidence_gates.append(
            (
                "debug_displayed_effective_cost_min_confidence",
                "displayed_effective_cost_confidence",
                "authoritative_live_displayed_effective_cost_confidence_unapproved",
            )
        )
    for threshold_name, confidence_name, stop_reason in confidence_gates:
        threshold = params.get(threshold_name)
        confidence = resource_transaction.get(confidence_name)
        confidence_gate_failed = bool(
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or float(confidence) < float(threshold)
        )
        if confidence_gate_failed:
            event["resource_transaction"] = resource_transaction
            event[threshold_name] = threshold
            preview_threshold = params.get("debug_preview_delta_min_confidence")
            preview_confidence = resource_transaction.get(
                "preview_delta_confidence"
            )
            preview_confidence_gate_approved = bool(
                not isinstance(preview_threshold, bool)
                and isinstance(preview_threshold, (int, float))
                and math.isfinite(float(preview_threshold))
                and 0.0 <= float(preview_threshold) <= 1.0
                and not isinstance(preview_confidence, bool)
                and isinstance(preview_confidence, (int, float))
                and math.isfinite(float(preview_confidence))
                and 0.0 <= float(preview_confidence) <= 1.0
                and float(preview_confidence) >= float(preview_threshold)
            )
            resource_retry_evidence = (
                _resource_absolute_confidence_retry_evidence(
                    resource_transaction,
                    resource_baseline,
                    threshold,
                )
                if (
                    stop_reason
                    == "authoritative_live_resource_absolute_confidence_unapproved"
                    and isinstance(resource_baseline, dict)
                )
                else None
            )
            if (resource_retry_evidence is None and getattr(context, 'gkh_hif_opening_state', None) is not None
                    and stop_reason == "authoritative_live_resource_absolute_confidence_unapproved"
                    and isinstance(resource_baseline, dict)
                    and resource_transaction.get('accepted') is True
                    and all(resource_transaction.get('conservation', {}).values())
                    and all(resource_transaction.get('normalized_'+name)==resource_baseline.get(name)
                            for name in ('stamina','shield'))):
                from live_decision.opening_observer import resource_baseline_reobserve_allowed
                candidate = dict(geometry_stable=event.get('unselected_geometry_authorized'),
                    select_consensus=event.get('select_consensus',{}),
                    resource_observations=resource_transaction.get('observations',{}).get('absolutes',{}).get('observations',()))
                if resource_baseline_reobserve_allowed(candidate, threshold):
                    resource_retry_evidence = dict(reason='hif_single_absolute_confidence_outlier')
            retryable_input_free_resource_absolute_a = bool(
                stop_reason
                == "authoritative_live_resource_absolute_confidence_unapproved"
                and preview_confidence_gate_approved
                and resource_retry_evidence is not None
                and allow_input_free_snapshot_retry is True
                and type(snapshot_retries_remaining) is int
                and snapshot_retries_remaining == 1
                and retry_index == 0
                and phase == "authoritative_refresh_a"
                and expected_selected_slot is None
                and expected_generation is None
                and input_click_counts_are_strict_zero
                and event["click_dispatched"] is False
            )
            if retryable_input_free_resource_absolute_a:
                assert input_click_counts is not None
                event.update(
                    {
                        "accepted": False,
                        "reason": stop_reason,
                        "snapshot_retry_triggered": True,
                        "snapshot_retry_kind": (
                            "input_free_authoritative_a_full_batch"
                        ),
                        "snapshot_retry_trigger": (
                            "resource_absolute_confidence"
                        ),
                        "snapshot_retry_evidence": resource_retry_evidence,
                        "snapshot_retries_remaining_before": (
                            snapshot_retries_remaining
                        ),
                        "input_click_counts": {
                            "identity_probe_click_count": input_click_counts[0],
                            "execution_selection_click_count": input_click_counts[1],
                            "execution_submit_count": input_click_counts[2],
                        },
                    }
                )
                try:
                    retry = _capture_verified_live_refresh(
                        context,
                        identity_certificate,
                        expected_selected_slot=expected_selected_slot,
                        expected_generation=expected_generation,
                        params=params,
                        plan=plan,
                        phase=phase,
                        recognizer=recognizer,
                        catalog=catalog,
                        policy=policy,
                        safety=safety,
                        expected_hand_count=expected_hand_count,
                        resource_baseline=resource_baseline,
                        selected_transition_verified=selected_transition_verified,
                        selected_identity_journal=selected_identity_journal,
                        allow_selected_score_preview_change=(
                            allow_selected_score_preview_change
                        ),
                        select_retries_remaining=0,
                        hand_count_retries_remaining=0,
                        allow_input_free_snapshot_retry=False,
                        snapshot_retries_remaining=0,
                        input_click_counts=input_click_counts,
                        retry_index=retry_index + 1,
                    )
                except LiveSelectionError as error:
                    raise LiveSelectionError(
                        str(error),
                        events=(event, *error.events),
                    ) from error
                retry["event"]["snapshot_retry_count"] = 1
                retry["event"]["snapshot_retries_remaining"] = 0
                retry["event"]["prior_attempts"] = [
                    event,
                    *list(retry["event"].get("prior_attempts", [])),
                ]
                return retry
            stop(stop_reason)
    observed_stamina = dict(fields["stamina"])
    observed_shield = dict(fields["shield"])
    fields["stamina"] = {
        "accepted": True,
        "value": resource_transaction["normalized_stamina"],
        "confidence": min(
            float(observed_stamina.get("confidence", 0.0)),
            float(resource_transaction["confidence"]),
        ),
        "reason": "selected_preview_normalized_to_preinput_baseline",
        "preview_value": resource_transaction.get("preview_stamina"),
        "preview_delta": resource_transaction.get("stamina_delta"),
    }
    fields["shield"] = {
        "accepted": True,
        "value": resource_transaction["normalized_shield"],
        "confidence": min(
            float(observed_shield.get("confidence", 0.0)),
            float(resource_transaction["confidence"]),
        ),
        "reason": "selected_preview_normalized_to_preinput_baseline",
        "preview_value": resource_transaction.get("preview_shield"),
        "preview_delta": resource_transaction.get("shield_delta"),
    }
    fields["resource_transaction"] = resource_transaction
    event["resource_transaction"] = resource_transaction
    event["visual_fields"] = fields
    try:
        generation = _live_generation_fingerprint(
            fields,
            hand_count=len(authoritative_candidates),
            plan=plan,
        )
    except LiveSelectionError as error:
        stop(str(error))
    event["generation"] = generation
    event["expected_generation"] = expected_generation
    event["generation_baseline_established"] = bool(
        phase == "authoritative_refresh_a" and expected_generation is None
    )
    if phase == "authoritative_refresh_b" and expected_generation is None:
        stop("authoritative_live_generation_baseline_missing")
    if expected_generation is not None and generation != expected_generation:
        expected_stable_generation = dict(expected_generation)
        current_stable_generation = dict(generation)
        expected_score = expected_stable_generation.pop("score", None)
        current_score = current_stable_generation.pop("score", None)
        score_preview_only = bool(
            "score" in expected_generation
            and "score" in generation
            and type(expected_score) is int
            and type(current_score) is int
            and current_score > expected_score
            and expected_stable_generation == current_stable_generation
        )
        if (
            phase == "authoritative_refresh_b"
            and allow_selected_score_preview_change
            and score_preview_only
        ):
            event["selected_score_preview_delta"] = {
                "expected": expected_score,
                "current": current_score,
                "delta": current_score - expected_score,
            }
        else:
            stop("authoritative_live_generation_changed")

    observed_cards: list[dict[str, Any]] = []
    for candidate in authoritative_candidates:
        certificate = certificate_by_slot[candidate.slot]
        fresh_prediction = fresh_by_slot[candidate.slot]
        entry = catalog.entries.get(str(certificate["card_id"]))
        if entry is None or entry.upgraded is not certificate["upgraded"]:
            stop("authoritative_live_catalog_identity_changed")
        current_title_confidence = None
        fresh_title_proof = event_proof(event, fresh_prediction)
        if fresh_title_proof is not None and fresh_title_proof.get('source_identity_kind') == 'inspected_art':
            from live_decision.hif_bound_title import valid_proof

            if valid_proof(context, fresh_prediction, fresh_title_proof):
                # Composite identity confidence comes from this group's title;
                # the weak visual prediction is retained unchanged in evidence.
                current_title_confidence = fresh_title_proof['confidence']
                revalidation_by_slot[candidate.slot]['complete_title_confidence'] = min(
                    float(certificate['probe_confidence']), current_title_confidence)
        confidence = _current_live_identity_confidence(certificate, fresh_prediction, event=event,
            selected_slot=expected_selected_slot, bound_hif=bound_continuation,
            minimum_confidence=params.get('minimum_identity_confidence'),
            current_title_confidence=current_title_confidence)
        margin = min(
            float(certificate["probe_margin"]),
            float(fresh_prediction.margin),
        )
        inspected_margin = None
        if (revalidation_by_slot[candidate.slot].get('acceptance_mode')
                == 'same_transaction_neighbor_exact_art'):
            inspected_margin = revalidation_by_slot[candidate.slot]['neighbor_continuity']
        if margin < float(params['debug_identity_min_margin']):
            from live_decision.hif_inspected_art import margin_evidence

            inspected_margin = inspected_margin or margin_evidence(
                context, certificate, fresh_prediction, event,
                float(params['debug_identity_min_margin']))
            if inspected_margin is not None:
                event.setdefault('inspected_title_margin_proofs', []).append(inspected_margin)
            else:
                from live_decision.hif_bound_title import margin_evidence as bound_margin_evidence

                inspected_margin = bound_margin_evidence(context, certificate, fresh_prediction,
                    event, float(params['debug_identity_min_margin']))
                if inspected_margin is not None:
                    event.setdefault('bound_title_margin_proofs', []).append(inspected_margin)
            if inspected_margin is None:
                from live_decision.hif_bound_title import selected_neighbor_margin_evidence

                inspected_margin = selected_neighbor_margin_evidence(
                    certificate, fresh_prediction, revalidation_by_slot[candidate.slot])
                if inspected_margin is not None:
                    event.setdefault('selected_neighbor_margin_proofs', []).append(inspected_margin)
            if inspected_margin is None:
                inspected_margin = _retained_hif_family_margin(context, certificate, fresh_prediction, event,
                    float(params['debug_identity_min_margin']))
        if (
            confidence < float(params["debug_identity_threshold"])
            or margin < float(params["debug_identity_min_margin"]) and inspected_margin is None
        ):
            stop("authoritative_live_identity_below_threshold")
        effective_cost = entry.stamina_cost
        cost_source = "catalog"
        if candidate.slot == expected_selected_slot:
            preview_value = resource_transaction.get("displayed_effective_cost")
            if (
                resource_transaction.get("accepted") is not True
                or type(preview_value) is not int
                or resource_transaction.get("selected_slot") != candidate.slot
                or resource_transaction.get("card_id") != entry.card_id
            ):
                stop("authoritative_live_resource_transaction_unbound")
            effective_cost = preview_value
            cost_source = "live_selected_card_face"
            preview_contract = resource_transaction.get('card_effect_preview') or {}
            if 'displayed_card_cost' in preview_contract:
                payment_cost = resource_transaction.get('effective_payment_cost')
                if (type(payment_cost) is not int or payment_cost != 0
                        or preview_contract.get('effective_cost') != payment_cost
                        or preview_contract.get('displayed_card_cost') != preview_value
                        or type(preview_contract.get('nullify_cost_cards')) is not int
                        or preview_contract['nullify_cost_cards'] <= 0):
                    stop('authoritative_live_nullified_payment_unbound')
                effective_cost = payment_cost
                cost_source = "live_selected_nullified_payment"
        rebound_prediction = replace(
            fresh_prediction,
            box=candidate.box,
            card_id=entry.card_id,
            predicted_card_id=entry.card_id,
            class_name=f"{entry.card_id}_{entry.name}",
            confidence=confidence,
            margin=margin,
            accepted=True,
            reason=(
                fresh_prediction.reason
                if fresh_prediction.reason in {
                    "live_selected_identity_certificate_carried",
                    "live_selected_right_neighbor_identity_carried",
                }
                else "live_identity_certificate_revalidated"
            ),
        )
        selected_gray_proof = None
        if (candidate.detector_label == 'hif_gray_observation'
                and candidate.slot == expected_selected_slot
                and getattr(context, 'gkh_hif_gray_selected_source', None) is not None):
            from live_decision.hif_gray_selection import selected_gray_playability
            try:
                selected_gray_proof = selected_gray_playability(
                    context.gkh_hif_opening_state,
                    complete_event=context.gkh_hif_committed_event,
                    journal=context.gkh_hif_gray_selected_source,
                    baseline=resource_baseline, event=event, candidate=candidate,
                    certificate=certificate,
                    geometry_comparator=_same_live_candidate_geometry)
            except ValueError as error:
                stop('authoritative_live_selected_gray_unproved:' + str(error))
            event['selected_gray_playability'] = selected_gray_proof
        observed_cards.append(
            {
                "prediction": rebound_prediction,
                "cost": effective_cost,
                "catalog_cost": entry.stamina_cost,
                "catalog_base_cost": entry.stamina_cost,
                "cost_source": cost_source,
                "playable": (candidate.detector_label in {"cards", "suggestions"}
                             or (isinstance(selected_gray_proof, dict)
                                 and selected_gray_proof.get('accepted') is True)),
                "features": catalog.features(
                    entry.card_id,
                    turns_remaining=int(fields["turns_remaining"]["value"]),
                ),
                "tags": [
                    f"plan:{plan}",
                    f"card_type:{entry.card_type}",
                    (
                        "identity:selected-unselected-certificate"
                        if rebound_prediction.reason
                        in {"live_selected_identity_certificate_carried",
                            "live_selected_right_neighbor_identity_carried"}
                        else "identity:visual-family-refresh"
                    ),
                    f"enhancement:{'plus' if entry.upgraded else 'normal'}",
                    *(["transaction:unsupported_typed_cost"]
                      if not entry.stamina_transaction_supported else []),
                    f"cost_source:{cost_source}",
                ],
                "upgraded": entry.upgraded,
                "upgrade_evidence": refresh_upgrade_evidence.get(
                    candidate.slot,
                    certificate.get("upgrade_evidence"),
                ),
            }
        )

    try:
        authorization_checked_at = _read_strict_live_capture_clock()
    except _LiveCaptureTimingError as error:
        event.update(error.details)
        stop(error.reason)
    if authorization_checked_at < captured_at:
        event.update(
            {
                "timing_failure": "clock_moved_backward_before_authorization",
                "authorization_checked_at": authorization_checked_at,
            }
        )
        stop("authoritative_live_capture_clock_invalid")
    capture_age_ms = (authorization_checked_at - captured_at) * 1000.0
    if not math.isfinite(capture_age_ms) or capture_age_ms < 0.0:
        event["timing_failure"] = "capture_age_not_finite_nonnegative"
        stop("authoritative_live_capture_clock_invalid")
    observation = {
        "snapshot_id": authoritative_frame_id,
        "capture_age_ms": capture_age_ms,
        "fields": fields,
        "card_predictions": observed_cards,
    }
    state = DmmLiveVisualStateProvider().build_state(observation)
    if callable(getattr(policy,'prepare_state',None)):
        try:
            state = policy.prepare_state(state, fields)
        except ValueError as error:
            stop('hif_full_state_revalidation_failed:'+str(error))
    proposal = policy.decide(state)
    decision = safety.filter(state, proposal)
    event.update(
        {
            "capture_age_ms": capture_age_ms,
            "input_deadline": input_deadline,
            "authorization_floor": authorization_checked_at,
            "proposal": proposal.to_dict(),
            "safe_decision": decision.to_dict(),
            "accepted": decision.has_action_intent,
            "reason": decision.reason,
        }
    )
    retryable_input_free_stale_a = bool(
        allow_input_free_snapshot_retry is True
        and type(snapshot_retries_remaining) is int
        and snapshot_retries_remaining == 1
        and retry_index == 0
        and phase == "authoritative_refresh_a"
        and expected_selected_slot is None
        and expected_generation is None
        and input_click_counts_are_strict_zero
        and event["click_dispatched"] is False
        and event["accepted"] is False
        and event["reason"] == "stale_state"
        and decision.has_action_intent is False
        and decision.reason == "stale_state"
        and capture_age_ms > max_snapshot_age_ms
    )
    if retryable_input_free_stale_a:
        assert input_click_counts is not None
        event.update(
            {
                "snapshot_retry_triggered": True,
                "snapshot_retry_kind": (
                    "input_free_authoritative_a_full_batch"
                ),
                "snapshot_retries_remaining_before": (
                    snapshot_retries_remaining
                ),
                "input_click_counts": {
                    "identity_probe_click_count": input_click_counts[0],
                    "execution_selection_click_count": input_click_counts[1],
                    "execution_submit_count": input_click_counts[2],
                },
            }
        )
        try:
            retry = _capture_verified_live_refresh(
                context,
                identity_certificate,
                expected_selected_slot=expected_selected_slot,
                expected_generation=expected_generation,
                params=params,
                plan=plan,
                phase=phase,
                recognizer=recognizer,
                catalog=catalog,
                policy=policy,
                safety=safety,
                expected_hand_count=expected_hand_count,
                resource_baseline=resource_baseline,
                selected_transition_verified=selected_transition_verified,
                selected_identity_journal=selected_identity_journal,
                allow_selected_score_preview_change=(
                    allow_selected_score_preview_change
                ),
                select_retries_remaining=0,
                hand_count_retries_remaining=0,
                allow_input_free_snapshot_retry=False,
                snapshot_retries_remaining=0,
                input_click_counts=input_click_counts,
                retry_index=retry_index + 1,
            )
        except LiveSelectionError as error:
            raise LiveSelectionError(
                str(error),
                events=(event, *error.events),
            ) from error
        retry["event"]["snapshot_retry_count"] = 1
        retry["event"]["snapshot_retries_remaining"] = 0
        retry["event"]["prior_attempts"] = [
            event,
            *list(retry["event"].get("prior_attempts", [])),
        ]
        return retry
    if phase == 'authoritative_refresh_a':
        _retain_hif_selected_neighbor_art(context, frames=frames, rows=frame_candidates,
            predictions=raw_fresh_predictions, decision=decision, generation=generation,
            selected_slot=expected_selected_slot, event=event, catalog=catalog)
    return {
        "captured_at": captured_at,
        "input_deadline": input_deadline,
        "authorization_floor": authorization_checked_at,
        "frames": frames,
        "candidates": authoritative_candidates,
        "current_selected_slot": select_consensus.selected_slot,
        "fields": fields,
        "generation": generation,
        "resource_transaction": resource_transaction,
        "card_predictions": observed_cards,
        "state": state,
        "proposal": proposal,
        "decision": decision,
        "event": event,
    }


def _select_live_target_for_refresh(
    context: Context,
    candidates: tuple[Any, ...],
    *,
    current_selected_slot: int | None,
    target_slot: int,
    input_deadline: float,
    authorization_floor: float,
    unselected_transition_authorized: bool = False,
    require_foreground_window: bool = False,
) -> tuple[bool, int, list[dict[str, Any]]]:
    """Select a different target once; a later authoritative refresh verifies it."""

    event = {
        "phase": "execution_selection",
        "attempted_slot": target_slot,
        "click_dispatched": False,
        "reason": None,
    }
    if current_selected_slot == target_slot:
        event.update({"accepted": True, "reason": "target_already_selected"})
        return True, 0, [event]
    if current_selected_slot is None and not unselected_transition_authorized:
        event.update(
            {
                "accepted": False,
                "reason": "unselected_transition_unverified",
            }
        )
        return False, 0, [event]
    if target_slot not in range(len(candidates)):
        event.update({"accepted": False, "reason": "target_slot_outside_hand"})
        return False, 0, [event]
    if not require_foreground_window:
        deadline_expired, observed_floor = _validate_live_input_clock_window(
            input_deadline,
            authorization_floor,
        )
        if deadline_expired:
            event.update(
                {"accepted": False, "reason": "snapshot_expired_before_input"}
            )
            return False, 0, [event]
        authorization_floor = observed_floor
    events = [event]
    if require_foreground_window:
        foreground_accepted, foreground_event = _debug_foreground_window_gate(
            input_deadline,
            authorization_floor=authorization_floor,
            allow_activation=False,
        )
        events.insert(0, foreground_event)
        if not foreground_accepted:
            event.update(
                {
                    "accepted": False,
                    "reason": foreground_event["activation_reason"],
                }
            )
            return False, 0, events
        foreground_floor = foreground_event.get("authorization_floor")
        if type(foreground_floor) in {int, float}:
            authorization_floor = float(foreground_floor)
    deadline_expired, observed_floor = _validate_live_input_clock_window(
        input_deadline,
        authorization_floor,
    )
    if deadline_expired:
        reason = (
            "snapshot_expired_after_foreground_gate"
            if require_foreground_window
            else "snapshot_expired_before_input"
        )
        if require_foreground_window:
            foreground_event.update(
                {
                    "accepted": False,
                    "activation_accepted": False,
                    "activation_reason": reason,
                }
            )
        event.update({"accepted": False, "reason": reason})
        return False, 0, events
    authorization_floor = observed_floor
    event["authorization_floor"] = authorization_floor
    try:
        dispatched = _click_live_candidate(
            context, _live_selection_click_candidate(candidates, target_slot), input_deadline=input_deadline,
            authorization_floor=authorization_floor,
        )
    except _LiveInputNotDispatched as error:
        event.update(accepted=False, reason=str(error),
                     input_may_have_been_dispatched=False)
        return False, 0, events
    except Exception as error:
        event.update(
            {
                "accepted": False,
                "reason": "selection_dispatch_exception",
                "error_type": type(error).__name__,
                "input_may_have_been_dispatched": True,
            }
        )
        return False, 1, events
    event["click_dispatched"] = dispatched
    if not dispatched:
        event.update({"accepted": False, "reason": "selection_click_failed"})
        return False, 1, events
    event.update({"accepted": True, "reason": "target_selection_dispatched"})
    time.sleep(SELECTION_SETTLE_SECONDS)
    return True, 1, events


def _submit_selected_live_card_once(
    context: Context,
    selected: Any,
    *,
    before_state: Any,
    input_deadline: float,
    authorization_floor: float,
) -> tuple[bool, int, list[dict[str, Any]]]:
    """Dispatch at most one business submit and then observe without retrying."""

    events: list[dict[str, Any]] = []
    submit_event = {
        "phase": "execution_submit",
        "attempted_slot": selected.slot,
        "click_dispatched": False,
    }
    deadline_expired, observed_floor = _validate_live_input_clock_window(
        input_deadline,
        authorization_floor,
    )
    if deadline_expired:
        submit_event["reason"] = "snapshot_expired_before_input"
        events.append(submit_event)
        return False, 0, events
    authorization_floor = observed_floor
    deadline_expired, observed_floor = _validate_live_input_clock_window(
        input_deadline,
        authorization_floor,
    )
    if deadline_expired:
        submit_event["reason"] = "snapshot_expired_before_input"
        events.append(submit_event)
        return False, 0, events
    submit_event["authorization_floor"] = observed_floor
    try:
        click_candidate = selected
        if (isinstance(getattr(context,'gkh_hif_live_context',None),HifLiveRunContext)
                and len(before_state.hand) in (4,5)):
            click_candidate = _live_selection_click_candidate(before_state.hand, selected.slot)
        dispatched = _click_live_candidate(
            context, click_candidate, input_deadline=input_deadline,
            authorization_floor=observed_floor,
        )
    except _LiveInputNotDispatched as error:
        submit_event.update(reason=str(error), input_may_have_been_dispatched=False)
        events.append(submit_event)
        return False, 0, events
    except Exception as error:
        submit_event.update(
            {
                "reason": "submit_dispatch_exception",
                "error_type": type(error).__name__,
                "input_may_have_been_dispatched": True,
            }
        )
        events.append(submit_event)
        return False, 1, events
    submit_event["click_dispatched"] = dispatched
    submit_event["reason"] = (
        "submit_dispatched" if dispatched else "submit_dispatch_failed"
    )
    events.append(submit_event)
    if not dispatched:
        return False, 1, events
    try:
        committed, postcondition_events = _confirm_live_selection_transition(
            context,
            selected,
            before_state,
        )
    except Exception as error:
        events.append(
            {
                "phase": "execution_postcondition",
                "attempted_slot": selected.slot,
                "click_dispatched": False,
                "accepted": False,
                "reason": "postcondition_exception",
                "error_type": type(error).__name__,
                "input_may_have_been_dispatched": True,
            }
        )
        return False, 1, events
    events.extend(postcondition_events)
    return committed, 1, events


def _unsubmitted_selected_b_stale(
    refresh, *, target_slot, expected_generation, recovered_selected, input_click_counts,
    selection_events=(),
):
    """Admit read-only reacquisition, never authorize the rejected snapshot."""
    event = refresh['event']
    proposal = refresh.get('proposal')
    action = getattr(proposal, 'action', None)
    if (not isinstance(input_click_counts, tuple) or len(input_click_counts) != 3
            or any(type(count) is not int for count in input_click_counts)):
        return False
    recovered = recovered_selected is True and input_click_counts == (0, 0, 0)
    selections = [step for step in selection_events if step.get('phase') == 'execution_selection']
    just_selected = (input_click_counts == (0, 1, 0) and len(selections) == 1
        and selections[0].get('accepted') is True
        and selections[0].get('reason') == 'target_selection_dispatched'
        and selections[0].get('attempted_slot') == target_slot
        and selections[0].get('click_dispatched') is True
        and all(step.get('phase') != 'execution_submit'
            and step.get('input_may_have_been_dispatched') is not True
            and (step is selections[0] or step.get('click_dispatched') is False)
            and not (isinstance(step.get('receipt'), dict) and step['receipt'].get('accepted') is False)
            for step in selection_events))
    return ((recovered or just_selected)
        and event.get('phase') == 'authoritative_refresh_b'
        and event.get('accepted') is False and event.get('reason') == 'stale_state'
        and event.get('click_dispatched') is False
        and refresh['decision'].has_action_intent is False
        and refresh['decision'].reason == 'stale_state'
        and refresh['generation'] == expected_generation
        and refresh['current_selected_slot'] == target_slot
        and getattr(proposal, 'status', None) == 'PROPOSE'
        and getattr(action, 'kind', None) == 'PLAY_CARD'
        and getattr(action, 'card_slot', None) == target_slot)


def _submit_selected_live_card_with_b_retry(
    context, refresh, *, target_slot, opening_policy, expected_generation,
    capture_again, refresh_events, recovered_selected=False, input_click_counts=None,
    selection_events=(),
):
    """Share one fresh B between an unsubmitted selection and final expiry."""
    events = []
    saved_bundle = getattr(opening_policy, 'bundle', None)
    saved_response = getattr(opening_policy, 'response', None)
    saved_action = getattr(saved_response, 'action', None)
    saved_instance = getattr(saved_action, 'card_instance_id', None)
    original_generation = dict(expected_generation)

    def binding_unchanged():
        return (saved_bundle is not None and saved_response is not None
            and opening_policy.bundle is saved_bundle and opening_policy.response is saved_response
            and saved_response.action is saved_action
            and getattr(saved_action, 'source_slot', None) == target_slot
            and type(target_slot) is int and 0 <= target_slot < len(saved_bundle.state.deck.hand)
            and saved_instance == saved_bundle.state.deck.hand[target_slot].instance_id)

    def generation_authorized(candidate):
        generation = candidate['generation']
        if generation == original_generation:
            return True
        event = candidate['event']
        expected_stable, current_stable = dict(original_generation), dict(generation)
        expected_score = expected_stable.pop('score', None)
        current_score = current_stable.pop('score', None)
        delta = event.get('selected_score_preview_delta')
        # Consume the complete B reader's existing score-preview exception;
        # this coordinator cannot create or broaden that observation proof.
        return (event.get('phase') == 'authoritative_refresh_b'
            and event.get('accepted') is True
            and event.get('reason') == 'all_safety_gates_passed'
            and event.get('allow_selected_score_preview_change') is True
            and event.get('expected_generation') == original_generation
            and event.get('generation') == generation
            and type(expected_score) is int and type(current_score) is int
            and current_score > expected_score and expected_stable == current_stable
            and isinstance(delta, dict) and all(type(value) is int for value in delta.values())
            and delta == dict(expected=expected_score, current=current_score,
                              delta=current_score-expected_score))

    # A renewed B must retain the exact first accepted forecast, including its
    # score. Passing the A-to-B exception twice cannot authorize a new value.
    accepted_generation = (dict(refresh['generation']) if generation_authorized(refresh)
                           else original_generation)

    def same_target(candidate):
        return (candidate['event'].get('accepted') is True
            and generation_authorized(candidate)
            and candidate['generation'] == accepted_generation
            and candidate['current_selected_slot'] == target_slot
            and candidate['decision'].has_action_intent
            and candidate['decision'].action.card_slot == target_slot)

    retry_bound = binding_unchanged() and same_target(refresh)
    for attempt in range(2):
        if not same_target(refresh):
            if not (attempt == 0 and binding_unchanged()
                    and _unsubmitted_selected_b_stale(refresh, target_slot=target_slot,
                        expected_generation=original_generation, recovered_selected=recovered_selected,
                        input_click_counts=input_click_counts, selection_events=selection_events)):
                if refresh['event'].get('accepted') is True:
                    events.append(dict(phase='execution_redecision_stop', attempted_slot=target_slot,
                        reason='selected_b_binding_unresolved', click_dispatched=False,
                        input_may_have_been_dispatched=False))
                return False, 0, events, refresh
            # The safety filter stops at age before some semantic gates. Do
            # not claim those gates passed or submit this rejected snapshot.
            retry_reason = ('initial_b_stale_after_selection' if input_click_counts == (0, 1, 0)
                            else 'initial_b_stale_without_input')
        else:
            selected = next((card for card in refresh['state'].hand
                             if card.slot == target_slot and card.box is not None), None)
            if selected is None:
                return False, 0, events, refresh
            executed, count, submitted = _submit_selected_live_card_once(context, selected,
                before_state=refresh['state'], input_deadline=refresh['input_deadline'],
                authorization_floor=refresh['authorization_floor'])
            events.extend(submitted)
            if not (attempt == 0 and retry_bound and binding_unchanged() and executed is False
                    and type(count) is int and count == 0 and len(submitted) == 1
                    and submitted[0].get('phase') == 'execution_submit'
                    and submitted[0].get('attempted_slot') == target_slot
                    and submitted[0].get('reason') == 'snapshot_expired_after_input_preparation'
                    and submitted[0].get('click_dispatched') is False
                    and submitted[0].get('input_may_have_been_dispatched') is False):
                return executed, count, events, refresh
            retry_reason = 'final_preparation_expired_without_dispatch'
        old_frames = refresh['event'].get('frame_ids', ())
        if (len(old_frames) != 3 or len(set(old_frames)) != 3
                or any(not isinstance(frame,str) or not frame for frame in old_frames)):
            return False, 0, events, refresh
        events.append(dict(phase='execution_submit_reobservation', attempted_slot=target_slot,
            reason=retry_reason, retry_count=1,
            click_dispatched=False, input_may_have_been_dispatched=False,
            prior_frame_ids=list(old_frames)))
        try:
            renewed = capture_again()
        except LiveSelectionError as error:
            raise LiveSelectionError(str(error), events=(*events,*error.events)) from error
        refresh_events.append(renewed['event'])
        new_frames = renewed['event'].get('frame_ids', ())
        if (not binding_unchanged() or not same_target(renewed)
                or len(new_frames) != 3 or len(set(new_frames)) != 3
                or any(not isinstance(frame,str) or not frame for frame in new_frames)
                or set(new_frames) & set(old_frames)):
            events.append(dict(phase='execution_redecision_stop', attempted_slot=target_slot,
                reason='fresh_b_retry_binding_unresolved', click_dispatched=False,
                input_may_have_been_dispatched=False))
            return False, 0, events, renewed
        refresh = renewed
    raise AssertionError('bounded B retry exhausted without returning')


def _infer_plan(context: Context) -> str:
    hif_run = getattr(context, "gkh_hif_live_context", None)
    if hif_run is not None:
        return hif_run.plan if isinstance(hif_run, HifLiveRunContext) else "free"
    node = context.get_node_data("ProduceChooseNIAEventFlag") or {}
    params = node.get("action", {}).get("param", {}).get("custom_action_param", {})
    effect = str(params.get("effect", "")) if isinstance(params, dict) else ""
    if effect.startswith("感性"):
        return "sense"
    if effect.startswith("理性"):
        return "logic"
    if effect.startswith("非凡"):
        return "anomaly"
    return "free"


def _locate_live_self_rank(context: Context, image: Any, *, candidates=None) -> dict[str, Any]:
    hif_run = getattr(context, "gkh_hif_live_context", None)
    if isinstance(hif_run, HifLiveRunContext):
        original = locate_hif_opening_self_rank(image,context=context)
        binding = getattr(context, 'gkh_hif_rank_preview_binding', None)
        if original['accepted'] or binding is None:
            return original
        from live_decision.hif_rank_preview import read_rank_preview

        if candidates is None:
            candidates = _detect_live_candidates(context, image)
        selected = _locate_hif_selected_slots(context, image, candidates, reuse_observation=True)
        return read_rank_preview(context, image, binding=binding,
            frame_id=frame_identifier(image), selected_slots=selected)
    if hif_run is not None:
        return {"accepted": False, "value": None, "confidence": 0.0,
                "reason": "invalid_hif_live_context"}
    return locate_dmm_self_rank(image)


def _capture_three_frames(context: Context) -> tuple[Any, Any, Any]:
    frames = []
    for index in range(3):
        frames.append(context.tasker.controller.post_screencap().wait().get())
        if index < 2:
            time.sleep(SELECTION_FRAME_INTERVAL_SECONDS)
    return tuple(frames)


def _capture_hif_panel_frames(context):
    """Read the current page once in the automatic battle loop."""
    return (context.tasker.controller.post_screencap().wait().get(),)


def _locate_selected_live_slots(
    context: Context,
    image: Any,
    candidates: tuple[Any, ...],
) -> tuple[int, ...]:
    """Locate SELECT labels only; yellow recommendation borders are ignored."""

    if isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext) and len(candidates) in (1, 2, 3, 4, 5):
        return _locate_hif_selected_slots(context, image, candidates)
    selected_slots = []
    for candidate in candidates:
        roi = live_select_label_box(tuple(image.shape), candidate)
        node_name = f"Task076SelectedLabel{candidate.slot}"
        detail = context.run_recognition(
            node_name,
            image,
            pipeline_override={
                node_name: {
                    "recognition": "OCR",
                    "roi": list(roi),
                    "expected": r"^SELECT$",
                }
            },
        )
        results = (
            list(getattr(detail, "filtered_results", ()) or ())
            if detail and detail.hit
            else []
        )
        if any(
            re.sub(r"[^A-Z]", "", str(getattr(result, "text", "")).upper())
            == "SELECT"
            for result in results
        ):
            selected_slots.append(candidate.slot)
    return tuple(selected_slots)


def _hif_select_plate_mask(crop):
    """Prove the existing unique orange label without assigning text."""
    import cv2
    import numpy as np

    if getattr(crop, 'shape', ()) != (48, 96, 3):
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, np.array([5,80,120],dtype=np.uint8), np.array([45,255,255],dtype=np.uint8))
    contours, _ = cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plates = [contour for contour in contours if cv2.contourArea(contour) >= 100]
    if len(plates) != 1:
        return None
    _, _, width, height = cv2.boundingRect(plates[0])
    if not (60 <= width <= 96 and 18 <= height <= 48
            and cv2.contourArea(plates[0]) >= width*height*.45):
        return None
    mask = np.zeros(orange.shape,dtype=np.uint8)
    cv2.drawContours(mask,plates,-1,255,-1)
    return mask


def _reread_hif_select_plate(context, crop, node, *, weak_results=None, unresolved_results=None):
    """Remove only pixels outside one orange label, then require exact OCR."""
    import numpy as np

    mask = _hif_select_plate_mask(crop)
    if mask is None:
        return None
    isolated = np.where(mask[:,:,None] > 0,crop,0).astype(crop.dtype)
    reread_node = node+'Plate'
    detail = context.run_recognition(reread_node,isolated,pipeline_override={reread_node: {
        'recognition':'OCR','roi':[0,0,96,48],'expected':r'^SELECT$','only_rec':True}})
    if unresolved_results is not None and detail is not None:
        unresolved_results.extend(getattr(detail, 'all_results', ()) or ())
    rows = list(getattr(detail,'all_results',()) or ()) if detail and detail.hit else []
    if (len(rows) == 1 and _valid_ocr_result(rows[0]) and str(rows[0].text).strip() == 'SELECT'
            and tuple(rows[0].box) == (0,0,96,48)):
        if rows[0].score >= .95:
            return rows[0]
        if weak_results is not None:
            weak_results.append(rows[0])
    return None


def _reread_hif_select_complete_border(context, crop, node):
    """Recognize a complete original label inside independently located whitespace."""
    import numpy as np

    if _hif_select_plate_mask(crop) is None:
        return None
    # The same one-slot whitespace layout as the proposal mosaic. No original
    # label pixel is masked, removed, resampled or supplied by another slot.
    padded = np.full((96,128,3),255,dtype=crop.dtype)
    padded[24:72,16:112] = crop
    detect_node = node+'CompleteBorder'
    detected = context.run_recognition(detect_node,padded,pipeline_override={detect_node: {
        'recognition':'OCR','roi':[0,0,128,96],'expected':r'^SELECT$'}})
    rows = list(getattr(detected,'all_results',()) or ()) if detected and detected.hit else []
    if (len(rows) != 1 or not _valid_ocr_result(rows[0])
            or str(rows[0].text).strip() != 'SELECT' or rows[0].score < .95):
        return None
    x,y,width,height = rows[0].box
    if (any(type(value) is not int for value in (x,y,width,height))
            or not (0 <= x <= 16 and 112 <= x+width <= 128
                    and 0 <= y <= 24 and 72 <= y+height <= 96)):
        return None
    # Detection only proposes a box. Require a second recognition-only read
    # whose entire crop still contains the original 96x48 source rectangle.
    complete = padded[y:y+height,x:x+width].copy()
    reread_node = node+'CompleteBorderText'
    detail = context.run_recognition(reread_node,complete,pipeline_override={reread_node: {
        'recognition':'OCR','roi':[0,0,width,height],'expected':r'^SELECT$','only_rec':True}})
    rows = list(getattr(detail,'all_results',()) or ()) if detail and detail.hit else []
    if (len(rows) == 1 and _valid_ocr_result(rows[0]) and str(rows[0].text).strip() == 'SELECT'
            and tuple(rows[0].box) == (0,0,width,height) and rows[0].score >= .95):
        return rows[0]
    return None


def _hif_select_fixed_glyph_proof(context, crop, original_rows):
    """Match all six complete glyphs; never replace the original OCR result."""
    import cv2
    import numpy as np

    if (getattr(crop, 'shape', ()) != (48,96,3) or crop.dtype != np.uint8
            or len(original_rows) != 1 or not _valid_ocr_result(original_rows[0])
            or tuple(original_rows[0].box) != (0,0,96,48)
            or not 0 < original_rows[0].score <= 1):
        return None
    # The calibrated white-core rule already used by HIF geometry. Ignore
    # only components touching the source border (outside the complete word).
    white = ((crop.min(2) >= 230) & (np.ptp(crop, axis=2) <= 20)).astype(np.uint8)
    _, _, components, _ = cv2.connectedComponentsWithStats(white, 8)
    glyphs = sorted((list(map(int,row)) for row in components[1:]
        if row[0] > 0 and row[1] > 0 and row[0]+row[2] < 96 and row[1]+row[3] < 48),
        key=lambda row: row[0])
    if len(glyphs) != 6:
        return None
    top = min(row[1] for row in glyphs)
    bottom = max(row[1]+row[3] for row in glyphs)
    height = bottom-top
    if (top < 2 or bottom > 46 or height not in (16,17)
            or any(a[0]+a[2] >= b[0] for a,b in zip(glyphs,glyphs[1:]))):
        return None
    original_text = str(original_rows[0].text).strip()
    if height == 16:
        if original_text != 'SELECT' or original_rows[0].score >= .95:
            return None
    elif _hif_select_plate_mask(crop) is None:
        return None
    mask = white[top-2:bottom+2]
    if mask[:2].any() or mask[-2:].any() or mask[:,:2].any() or mask[:,-2:].any():
        return None
    # Retain the full source width, every glyph pixel and two blank white-core
    # rows above/below; no resampling or orange-background-only match.
    strip = crop[top-2:bottom+2].copy()
    template_name = ('hif_select_complete_glyphs.png' if height == 16
                     else 'hif_select_complete_glyphs_1841.png')
    glyph_confidences = []
    if height == 17:
        # The current raster's complete S/C are one pixel taller. Its fixed
        # reference is an independently strong SELECT from 1841, retaining
        # every original pixel. Match each letter as well as the whole word;
        # orange background or five matching letters cannot hide a wrong one.
        reference = cv2.imread(str(_resolve_runtime_path(
            Path('assets/resource/base/image/produce')/template_name,
            Path('resource/base/image/produce')/template_name)))
        if reference is None or reference.shape != strip.shape:
            return None
        for x,_y,width,_height,_area in glyphs:
            left,right = x-2,x+width+2
            if not 0 <= left < right <= 96:
                return None
            score = float(cv2.matchTemplate(strip[:,left:right],reference[:,left:right],
                                           cv2.TM_CCOEFF_NORMED)[0,0])
            if not .95 <= score <= 1:
                return None
            glyph_confidences.append(score)
    node = 'Task275HifSelectFixedGlyphs'
    detail = context.run_recognition(node, strip, pipeline_override={node: dict(
        recognition='TemplateMatch', template=['produce/'+template_name],
        roi=[0,0,96,height+4], threshold=.95)})
    rows = list(getattr(detail,'filtered_results',()) or ()) if detail and detail.hit else []
    if (len(rows) != 1 or tuple(rows[0].box) != (0,0,96,height+4)
            or type(rows[0].score) not in (int,float) or not .95 <= rows[0].score <= 1):
        return None
    proof = dict(source=('native_1658_complete_select_glyph_template' if height == 16
                         else 'native_1841_complete_select_glyph_template'),
        confidence=float(rows[0].score), glyph_roi=[0,top-2,96,height+4], glyph_components=glyphs,
        original_ocr=dict(text=str(original_rows[0].text), confidence=float(original_rows[0].score),
                          box=list(original_rows[0].box)))
    if height == 17:
        proof['glyph_confidences'] = glyph_confidences
    return proof


def _hif_select_complete_glyph_band(context, crop, original_rows, node):
    """Read one complete word band; outside stage glints cannot become letters."""
    import cv2
    import numpy as np

    if (getattr(crop,'shape',()) != (48,96,3) or crop.dtype != np.uint8
            or len(original_rows) != 1 or not _valid_ocr_result(original_rows[0])
            or tuple(original_rows[0].box) != (0,0,96,48)
            or not 0 < original_rows[0].score <= 1):
        return None
    plate = _hif_select_plate_mask(crop)
    if plate is None:
        return None
    white = ((crop.min(2)>=230)&(np.ptp(crop,axis=2)<=20)).astype(np.uint8)
    _,_,stats,_ = cv2.connectedComponentsWithStats(white,8)
    components = [list(map(int,row)) for row in stats[1:]
        if row[0]>0 and row[1]>0 and row[0]+row[2]<96 and row[1]+row[3]<48]
    # Flat-top E/L may have a 15px white core while rounded S/C span 16px.
    # Keep the existing complete word height and all six separate glyphs.
    glyphs = sorted((row for row in components if 15<=row[3]<=17),key=lambda row:row[0])
    if len(glyphs) != 6:
        return None
    top,bottom = min(row[1] for row in glyphs),max(row[1]+row[3] for row in glyphs)
    px,py,pw,ph = cv2.boundingRect(plate)
    if (top<2 or bottom>46 or bottom-top not in (16,17)
            or any(a[0]+a[2]>=b[0] for a,b in zip(glyphs,glyphs[1:]))
            # The pulsing orange edge can recede behind a glyph's outer
            # stroke. Each complete glyph still belongs to this plate by
            # its centre, and the untouched word receives independent OCR.
            or any(not (px<=x+w/2<px+pw and py<=y and y+h<=py+ph)
                   for x,y,w,h,_area in glyphs)
            or any(row not in glyphs and row[1]<bottom+2 and row[1]+row[3]>top-2
                   for row in components)):
        return None
    word_left,word_right = glyphs[0][0]-2,glyphs[-1][0]+glyphs[-1][2]+2
    if not 0<=word_left<word_right<=96:
        return None
    band_white = white[top-2:bottom+2,word_left:word_right]
    if (band_white[:2].any() or band_white[-2:].any()
            or band_white[:,:2].any() or band_white[:,-2:].any()):
        return None
    # Keep all 96 source columns, all six original glyphs and the independently
    # blank rows. Use the existing one fixed 2x OCR transform, never a scale search.
    band = crop[top-2:bottom+2].copy()
    enlarged = cv2.resize(band,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    roi = [0,0,192,len(band)*2]
    read_node = node+'CompleteGlyphBand'
    detail = context.run_recognition(read_node,enlarged,pipeline_override={read_node:dict(
        recognition='OCR',roi=roi,expected=r'^SELECT$',only_rec=True)})
    rows = list(getattr(detail,'all_results',()) or ()) if detail and detail.hit else []
    if (len(rows)!=1 or not _valid_ocr_result(rows[0]) or str(rows[0].text).strip()!='SELECT'
            or tuple(rows[0].box)!=tuple(roi) or not .95<=rows[0].score<=1):
        return None
    return dict(source='complete_select_glyph_band_fixed_double_ocr',confidence=rows[0].score,
        glyph_roi=[0,top-2,96,len(band)],glyph_components=glyphs,scale=2,
        plate_box=[px,py,pw,ph],original_ocr=dict(text=str(original_rows[0].text),
            confidence=original_rows[0].score,box=list(original_rows[0].box)))


def _hif_select_complete_white_glyphs(context, crop, original_rows, node):
    """Separate complete enclosed white letters from the small-window glow."""
    import cv2
    import numpy as np

    if (getattr(context, 'gkh_hif_client_size', None) not in ((540,960),(541,961),(541,962),(542,963))
            or getattr(crop, 'shape', ()) != (48,96,3) or crop.dtype != np.uint8
            or len(original_rows) != 1 or not _valid_ocr_result(original_rows[0])
            or tuple(original_rows[0].box) != (0,0,96,48)):
        return None
    plate = _hif_select_plate_mask(crop)
    if plate is None:
        return None
    white = ((crop.min(2)>=230)&(np.ptp(crop,axis=2)<=20)).astype(np.uint8)
    _,labels,stats,_ = cv2.connectedComponentsWithStats(white,8)
    enclosed = [(index,list(map(int,row))) for index,row in enumerate(stats[1:],1)
        if row[0]>0 and row[1]>0 and row[0]+row[2]<96 and row[1]+row[3]<48]
    glyphs = sorted(((index,row) for index,row in enclosed if 15<=row[3]<=17),
        key=lambda pair:pair[1][0])
    if len(glyphs) != 6:
        return None
    top,bottom = min(row[1] for _,row in glyphs),max(row[1]+row[3] for _,row in glyphs)
    px,py,pw,ph = cv2.boundingRect(plate)
    if (top<2 or bottom>46 or bottom-top not in (16,17)
            or any(a[0]+a[2]>=b[0] for (_,a),(_,b) in zip(glyphs,glyphs[1:]))
            or any(not (px<=x+w/2<px+pw and py<=y and y+h<=py+ph)
                   for _,(x,y,w,h,_area) in glyphs)
            or any((index,row) not in glyphs and row[1]<bottom and row[1]+row[3]>top-2
                   for index,row in enclosed)):
        return None
    # Every retained pixel belongs to one complete source component. Never
    # synthesize letters or remove an enclosed accent/extra stroke in the word.
    ink = np.isin(labels,[index for index,_ in glyphs])
    text = 255-ink[top-2:bottom+2].astype(np.uint8)*255
    enlarged = cv2.resize(cv2.cvtColor(text,cv2.COLOR_GRAY2BGR),None,
        fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    roi = [0,0,192,len(enlarged)]
    read_node = node+'CompleteWhiteGlyphs'
    detail = context.run_recognition(read_node,enlarged,pipeline_override={read_node:dict(
        recognition='OCR',roi=roi,expected=r'^SELECT$',only_rec=True)})
    rows = list(getattr(detail,'all_results',()) or ()) if detail and detail.hit else []
    if (len(rows)!=1 or not _valid_ocr_result(rows[0]) or str(rows[0].text).strip()!='SELECT'
            or tuple(rows[0].box)!=tuple(roi) or not .95<=rows[0].score<=1):
        return None
    return dict(source='complete_select_enclosed_white_glyphs',confidence=rows[0].score,
        glyph_roi=[0,top-2,96,len(text)],glyph_components=[row for _,row in glyphs],scale=2,
        plate_box=[px,py,pw,ph],original_ocr=dict(text=str(original_rows[0].text),
            confidence=original_rows[0].score,box=list(original_rows[0].box)))


def _locate_upstream_hif_selected_slots(context, image, candidates):
    """Use the same pinned SELECT/panel/raised-card reader as upstream input."""
    from live_decision.upstream_hif_card_ui import BattleRecognitionMixin

    ui = BattleRecognitionMixin()
    ui._reset_hif_recognition()
    boxes = [(list(card.box), card.detector_score) for card in candidates]
    ui._hif_bind_hand(boxes)
    selected = ui._hif_selected_index(context, image, boxes, pending=False)
    if selected == -2:
        raise LiveSelectionError('upstream HIF selection is unresolved')
    return () if selected == -1 else (selected,)


def _locate_hif_selected_slots(context, image, candidates, *, reuse_observation=False) -> tuple[int, ...]:
    if getattr(context, 'gkh_hif_upstream_card_ui', False) is True:
        return _locate_upstream_hif_selected_slots(context, image, candidates)
    try:
        rois = hif_select_label_rois(tuple(image.shape), candidates)
    except (TypeError, ValueError) as error:
        raise LiveSelectionError("HIF selection geometry invalid") from error
    cache_key = (frame_identifier(image), _live_candidate_signature(candidates), tuple(rois))
    cached = getattr(context, "gkh_hif_select_cache", ())
    if not isinstance(cached, tuple):
        cached = ()
    if reuse_observation:
        for key, slots, *_ in cached:
            if key == cache_key:
                return slots
    visual_proofs = []
    proposal_slots = range(len(rois))
    if len(rois) in (3, 4, 5):
        import numpy as np

        # Keep original pixels and isolate each label with whitespace. Detection
        # proposes slots only; the original recognition-only check proves them.
        mosaic_width = 128 * len(rois)
        mosaic = np.full((96, mosaic_width, 3), 255, dtype=image.dtype)
        for slot, (x, y, width, height) in enumerate(rois):
            mosaic[24:24+height, slot*128+16:slot*128+16+width] = image[y:y+height, x:x+width]
        node = "Task275HifSelectMosaic"
        detail = context.run_recognition(node, mosaic, pipeline_override={node: {
            "recognition": "OCR", "roi": [0, 0, mosaic_width, 96], "expected": r"^SELECT$",
        }})
        proposals = list(getattr(detail, "filtered_results", ()) or ()) if detail and detail.hit else []
        slots = []
        for result in proposals:
            if not _valid_ocr_result(result) or str(result.text).strip() != "SELECT":
                raise LiveSelectionError("HIF selection returned invalid evidence")
            x, y, width, height = result.box
            slot = int((x + width / 2) // 128)
            if not (0 <= slot < len(rois) and 0 <= x < x+width <= mosaic_width
                    and 0 <= y < y+height <= 96) or slot in slots:
                raise LiveSelectionError("HIF selection target is ambiguous")
            if not slot*128 <= x < x+width <= (slot+1)*128:
                # OCR padding can cross a cell's blank margin. Only the
                # original pasted source rectangles determine ownership;
                # the unchanged full-source reread below proves SELECT.
                owners = [i for i, (_rx, _ry, rw, rh) in enumerate(rois)
                          if x < i*128+16+rw and x+width > i*128+16
                          and y < 24+rh and y+height > 24]
                if owners != [slot]:
                    raise LiveSelectionError("HIF selection target is ambiguous")
            slots.append(slot)
        # The text detector can miss a pulsing label. Raised geometry may
        # request the same strict OCR reread, never authorize selection itself.
        geometry = _live_selection_geometry_evidence(candidates)
        proposal_slots = tuple(sorted({*slots, *geometry["raised_slots"]}))
    selected = []
    for slot in proposal_slots:
        roi = rois[slot]
        node = f"Task275HifSelect{slot}"
        x, y, width, height = roi
        crop = image[y:y+height, x:x+width].copy()
        crop_roi = [0, 0, width, height]
        detail = context.run_recognition(node, crop, pipeline_override={node: {
            "recognition": "OCR", "roi": crop_roi, "expected": r"^SELECT$",
            "only_rec": True,
        }})
        results = list(getattr(detail, "filtered_results", ()) or ()) if detail and detail.hit else []
        original_rows = list(getattr(detail, 'all_results', ()) or ()) if detail else []
        complete_border_result = None
        weak_plate_results = []
        unresolved_plate_results = []
        proof = (_hif_select_fixed_glyph_proof(context,crop,original_rows)
            if _live_selection_geometry_evidence(candidates)['raised_slots'] == (slot,) else None)
        if (proof is None and (not results or len(results)==1 and results[0].score<.95)
                and _live_selection_geometry_evidence(candidates)['raised_slots'] == (slot,)):
            proof = _hif_select_complete_white_glyphs(context,crop,original_rows,node)
        if proof is not None:
            visual_proofs.append(dict(proof, frame_id=cache_key[0], slot=slot, roi=list(roi)))
            selected.append(slot)
            continue
        if ((not results or len(results) == 1 and results[0].score < .95)
                and len(original_rows) == 1 and _valid_ocr_result(original_rows[0])
                # The original OCR only requests an independent orange-plate
                # reread. Its spelling is never an acceptance credential.
                and str(original_rows[0].text).strip().lstrip("\"'") in ('SELECT', 'SÉLECT')
                and _live_selection_geometry_evidence(candidates)['raised_slots'] == (slot,)):
            isolated_result = _reread_hif_select_plate(context,crop,node,weak_results=weak_plate_results,
                unresolved_results=unresolved_plate_results)
            if isolated_result is not None:
                results = [isolated_result]
            elif weak_plate_results or str(original_rows[0].text).strip() == 'SELECT':
                complete_border_result = _reread_hif_select_complete_border(context,crop,node)
                if complete_border_result is not None:
                    results = [complete_border_result]
        if ((not results or len(results) == 1 and results[0].score < .95)
                and _live_selection_geometry_evidence(candidates)['raised_slots'] == (slot,)):
            proof = _hif_select_complete_glyph_band(context,crop,original_rows,node)
            if proof is not None:
                visual_proofs.append(dict(proof,frame_id=cache_key[0],slot=slot,roi=list(roi),
                    plate_ocr=[dict(text=str(row.text),confidence=row.score,box=list(row.box))
                               for row in unresolved_plate_results]))
                selected.append(slot)
                continue
        if not results:
            if (len(original_rows) == len(unresolved_plate_results) == 1
                    and _valid_ocr_result(original_rows[0]) and _valid_ocr_result(unresolved_plate_results[0])
                    and crop.shape == (48,96,3)
                    and _live_selection_geometry_evidence(candidates)['raised_slots'] == (slot,)):
                # Neither OCR nor the unchanged six-glyph proof accepted this
                # label. Preserve both reads solely for a fresh whole batch.
                unresolved = dict(
                    phase='hif_select_label', reason='select_label_accent_unresolved', accepted=False,
                    slot=slot, raised_slots=[slot], frame_id=cache_key[0], input_click_count=0,
                    original_ocr=dict(text=getattr(original_rows[0], 'text', None), score=original_rows[0].score,
                        box=list(original_rows[0].box)),
                    plate_ocr=dict(text=getattr(unresolved_plate_results[0], 'text', None), score=unresolved_plate_results[0].score,
                        box=list(unresolved_plate_results[0].box)))
                if _unresolved_select_label_diagnostic(unresolved, slot):
                    raise LiveSelectionError('HIF raised card has unresolved SELECT', events=[unresolved])
            if (len(weak_plate_results)==1 and len(original_rows)==1
                    and str(original_rows[0].text).strip()=='SÉLECT' and original_rows[0].score<.95):
                # 1643: the independently isolated complete plate did read
                # exact SELECT, but below the unchanged .95 gate. Preserve
                # this rejection for the existing one-batch reobservation;
                # the accented original is never accepted as SELECT.
                raise LiveSelectionError("HIF SELECT confidence below threshold", events=[{
                    "phase": "hif_select_label", "reason": "exact_select_low_confidence",
                    "slot": slot, "confidence": weak_plate_results[0].score,
                    "frame_id": frame_identifier(image), "input_click_count": 0,
                    "source": "isolated_complete_select_plate",
                    "original_text": str(original_rows[0].text),
                    "original_confidence": original_rows[0].score}])
            continue
        if len(results) != 1:
            raise LiveSelectionError("HIF selection target is ambiguous")
        result = results[0]
        if (not _valid_ocr_result(result)
                or str(getattr(result, "text", "")).strip() != "SELECT"
                or result is not complete_border_result and tuple(result.box) != tuple(crop_roi)):
            raise LiveSelectionError("HIF selection returned invalid evidence")
        if (result.score < .95
                and getattr(context, 'gkh_hif_gray_selected_source', None) is not None):
            import cv2

            enlarged = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            enlarged_roi = [0, 0, width*2, height*2]
            enlarged_node = node + 'Magnified'
            enlarged_detail = context.run_recognition(enlarged_node, enlarged,
                pipeline_override={enlarged_node: {
                    'recognition': 'OCR', 'roi': enlarged_roi,
                    'expected': r'^SELECT$', 'only_rec': True}})
            enlarged_results = (list(getattr(enlarged_detail, 'filtered_results', ()) or ())
                                if enlarged_detail and enlarged_detail.hit else [])
            if (len(enlarged_results) == 1 and _valid_ocr_result(enlarged_results[0])
                    and str(enlarged_results[0].text).strip() == 'SELECT'
                    and tuple(enlarged_results[0].box) == tuple(enlarged_roi)
                    and enlarged_results[0].score >= .95):
                result = enlarged_results[0]
        if result.score < .95:
            raise LiveSelectionError("HIF SELECT confidence below threshold", events=[{
                "phase": "hif_select_label", "reason": "exact_select_low_confidence",
                "slot": slot, "confidence": result.score, "frame_id": frame_identifier(image),
                "input_click_count": 0}])
        selected.append(slot)
    tops = sorted(card.box[1] for card in candidates)
    if not selected and (tops[min(2,len(tops)-1)] - tops[0] >= 20
                         or len(tops)==1 and tops[0]<878):
        # A raised card with unreadable SELECT must never become permission to
        # click that card as though it were unselected (which would submit it).
        raise LiveSelectionError("HIF raised card has unresolved SELECT")
    context.gkh_hif_select_cache = (*tuple(item for item in cached if item[0] != cache_key)[-2:],
                                    (cache_key, tuple(selected), tuple(visual_proofs)))
    return tuple(selected)


def _infer_selected_live_slot_from_geometry(
    candidates: tuple[Any, ...],
) -> int | None:
    """Return one SELECT-expanded slot from a calibrated card row."""

    return _live_selection_geometry_evidence(candidates)["selected_slot"]


def _observe_expected_selected_live_slots(
    context: Context,
    image: Any,
    candidates: tuple[Any, ...],
    expected_selected_slot: int | None,
    *,
    ocr_slots: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Fuse label OCR with independent calibrated expansion geometry."""

    if getattr(context, 'gkh_hif_upstream_card_ui', False) is True:
        slots = _locate_upstream_hif_selected_slots(context, image, candidates)
        raised = [card.slot for card in candidates
            if card.box[1] <= 875 and card.box[1] + card.box[3] <= 1130]
        return slots, dict(source='pinned_upstream_select_panel_and_raised_card',
            ocr_slots=list(slots), geometry_available=True, geometry_ambiguous=False,
            geometry_raised_slots=raised, geometry_slot=raised[0] if len(raised) == 1 else None,
            used_geometry_fallback=False, resolved_slots=list(slots), frame_id=frame_identifier(image))
    if ocr_slots is None:
        ocr_slots = (
            _locate_hif_selected_slots(context, image, candidates, reuse_observation=True)
            if isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext)
            and len(candidates) in (1, 2, 3, 4, 5)
            else _locate_selected_live_slots(context, image, candidates)
        )
    geometry = _live_selection_geometry_evidence(candidates)
    geometry_slot = geometry["selected_slot"]
    resolved_slots = ocr_slots
    used_geometry_fallback = False
    if geometry["ambiguous"]:
        possible_slots = geometry["raised_slots"] or tuple(
            candidate.slot for candidate in candidates
        )
        resolved_slots = tuple(sorted({*ocr_slots, *possible_slots}))
    elif ocr_slots and geometry_slot is not None and geometry_slot not in ocr_slots:
        resolved_slots = tuple(sorted({*ocr_slots, geometry_slot}))
    elif not ocr_slots and geometry_slot is not None and (
        expected_selected_slot is None or geometry_slot == expected_selected_slot
    ):
        resolved_slots = (geometry_slot,)
        used_geometry_fallback = True
    evidence = {
        "ocr_slots": list(ocr_slots),
        "geometry_available": geometry["available"],
        "geometry_ambiguous": geometry["ambiguous"],
        "geometry_raised_slots": list(geometry["raised_slots"]),
        "geometry_slot": geometry_slot,
        "resolved_slots": list(resolved_slots),
        "used_geometry_fallback": used_geometry_fallback,
    }
    cached = getattr(context, 'gkh_hif_select_cache', ())
    if isinstance(cached, tuple) and cached:
        current = (frame_identifier(image), _live_candidate_signature(candidates))
        for key, _, *proofs in cached:
            if key[:2] == current and proofs and proofs[0]:
                evidence['select_visual_proofs'] = [dict(proof) for proof in proofs[0]]
    return resolved_slots, evidence


def _normalise_observed_live_candidates(context, image, results):
    upstream_geometry = getattr(context, 'gkh_hif_upstream_card_ui', False) is True
    if upstream_geometry:
        from live_decision.upstream_hif_card_ui import BattleRecognitionMixin

        # Apply the exact upstream normal-card geometry filter. Non-playable
        # slots remain business observations, never extra upstream click targets.
        rows = BattleRecognitionMixin()._hif_card_boxes(results)
        allowed = {tuple(box) for box, _ in rows}
        results = [row for row in results if row.label == 'useless' or tuple(row.box) in allowed]
    # HIF's selected preview can dim neighbouring cards. They still occupy
    # physical hand slots; preserve their original non-playable label.
    raw = isolate_card_candidates(results, include_unplayable=isinstance(
        getattr(context, "gkh_hif_live_context", None), HifLiveRunContext,
    ))
    if upstream_geometry:
        # The pinned upstream owns physical geometry on this explicit path.
        # Legacy title-plate repair rejects valid adjacent gray cards even
        # when the upstream detector retains their complete row. SELECT,
        # identity, three-frame stability and business-state checks follow.
        normalized = normalise_live_candidates(raw, frame_shape=getattr(image, 'shape', None))
        if any(c.detector_label == 'suggestions' and c.box[3] >= 270 for c in normalized):
            from live_decision.hif_visual import _hif_complete_halo_title_matches

            # Reuse the existing, independent title-owned halo proof without
            # requiring every gray neighbour to also expose a title plate.
            plates = HifFrameTitleCache(image).unselected(image)
            repaired = []
            for card in normalized:
                matches = (_hif_complete_halo_title_matches(card, plates)
                    if card.detector_label == 'suggestions' and card.box[3] >= 270 else ())
                repaired.append(replace(card, box=matches[0]) if len(matches) == 1 else card)
            normalized = normalise_live_candidates(repaired, frame_shape=tuple(image.shape))
        if normalized:
            count = len(normalized)
            spacing = 218 if count <= 3 else 494 / (count - 1)
            origin = (720 - 190 - (count - 1) * spacing) / 2
            # A missing gray card must not turn a four-slot row into an
            # apparently stable three-card hand. Prove each physical slot
            # against the calibrated row, independently of expected identities.
            if any(not round(origin + slot * spacing) - 8 <= card.box[0]
                    <= round(origin + slot * spacing) + 190 - card.box[2] + 8
                    for slot, card in enumerate(normalized)):
                return ()
        return normalized
    is_hif = isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext)
    if is_hif:
        title_cache = HifFrameTitleCache(image)
        for hand_count in (5, 4, 3, 2, 1):
            selected_row = hif_selected_row_proposal(image, raw, hand_count=hand_count, title_cache=title_cache)
            if selected_row is not None:
                repaired, selected_slot = selected_row
                if not repaired or _locate_hif_selected_slots(context, image, repaired) != (selected_slot,):
                    return ()
                return normalise_live_candidates(repaired, frame_shape=tuple(image.shape))
        unselected = hif_unselected_row_candidates(tuple(image.shape), raw, image=image, title_cache=title_cache)
        if unselected is not None:
            return normalise_live_candidates(unselected, frame_shape=tuple(image.shape))
    normalized = normalise_live_candidates(raw, frame_shape=getattr(image, "shape", None))
    has_fragment = any(c.box[1] >= 704 and c.box[2] < 105 for c in raw)
    if not is_hif or (normalized and not has_fragment):
        return normalized
    proposal = hif_selected_fragment_proposal(tuple(image.shape), raw)
    if proposal is None:
        return ()
    repaired, selected_slot = proposal
    # Geometry alone cannot remove a detector result. Require an exact SELECT
    # label on this very frame, mapped to the merged card and no other slot.
    if _locate_hif_selected_slots(context, image, repaired) != (selected_slot,):
        return ()
    return normalise_live_candidates(repaired, frame_shape=tuple(image.shape))


def _detect_live_candidates(context: Context, image: Any) -> tuple[Any, ...]:
    detail = context.run_recognition("ProduceRecognitionCards", image)
    results = (
        list(getattr(detail, "all_results", ()) or ())
        if detail and detail.hit
        else []
    )
    return _normalise_observed_live_candidates(context, image, results)


def _reconcile_hif_unselected_detector_gap(context, frames, rows):
    """Use two current rows to locate a missing row, then prove its current pixels."""
    if (not isinstance(getattr(context, 'gkh_hif_live_context', None), HifLiveRunContext)
            or len(frames) != 3 or len(rows) != 3
            or getattr(context, 'gkh_hif_gray_selected_source', None) is not None):
        return rows, None
    present = [i for i, row in enumerate(rows) if row]
    if len(present) != 2 or not _same_live_candidate_geometry(rows[present[0]], rows[present[1]]):
        return rows, None
    frame_ids = tuple(frame_identifier(frame) for frame in frames)
    if len(set(frame_ids)) != 3:
        return rows, None
    event = getattr(context, 'gkh_hif_committed_event', None) or {}
    bundle = getattr(context, 'gkh_hif_opening_state', None)
    if (bundle is None or event.get('event') != 'live_committed_state'
            or event.get('expected_state', {}).get('turn') != bundle.state.turn
            or tuple(event.get('core', {}).get('fields', ())[:4]) != (
                bundle.state.turns_remaining, bundle.state.stamina, bundle.state.genki, bundle.state.score)):
        return rows, None
    try:
        # Identity remains the already verified business hand. This reference
        # is only for this batch's physical row, never an old input coordinate.
        identities = _retained_hif_hand_ids(context, tuple(map(tuple, event['core']['families'])))
    except (KeyError, TypeError, ValueError, AttributeError, LiveSelectionError):
        return rows, None
    reference = rows[present[-1]]
    if len(identities) != len(reference):
        return rows, None
    missing = next(i for i, row in enumerate(rows) if not row)
    image = frames[missing]
    projected = hif_unselected_row_candidates(tuple(image.shape), reference, image=image)
    if not projected or not _same_live_candidate_geometry(reference, projected):
        return rows, None
    # These boxes are located by current title/body pixels. Do not pass off
    # another frame's detector score as a new neural observation.
    projected = tuple(replace(card, detector_score=0.) for card in projected)
    repaired = tuple(projected if i == missing else row for i, row in enumerate(rows))
    return repaired, dict(phase='same_batch_unselected_geometry', frame_id=frame_ids[missing],
        reference_frame_ids=[frame_ids[i] for i in present], card_ids=list(identities),
        original_candidate_boxes=[[list(c.box) for c in row] for row in rows],
        current_pixel_candidate_boxes=[list(c.box) for c in projected], input_click_count=0)


def _observe_live_selected_slot_consensus(
    context: Context,
    frames: tuple[Any, ...],
    *,
    expected_selected_slot: int | None,
    first_frame_candidates: tuple[Any, ...] | None = None,
    reconciliation_events: list[dict[str, Any]] | None = None,
    expected_hand_count: int | None = None,
) -> tuple[Any | None, tuple[tuple[Any, ...], ...], list[dict[str, Any]], bool]:
    """Locate current physical rows before accepting label/geometry SELECT evidence."""

    current_frame = len(frames) == 1 and getattr(context, 'gkh_hif_upstream_card_ui', False) is True
    if len(frames) != 3 and not current_frame:
        raise LiveSelectionError("Live SELECT observation requires exactly three frames")
    frame_candidates = tuple(
        first_frame_candidates
        if index == 0 and first_frame_candidates is not None
        else _detect_live_candidates(context, frame)
        for index, frame in enumerate(frames)
    )
    frame_candidates, fragment_reconciliation = (
        _reconcile_single_live_detector_fragment(
            frame_candidates,
            expected_hand_count=expected_hand_count,
        )
    )
    if fragment_reconciliation is not None and reconciliation_events is not None:
        reconciliation_events.append(fragment_reconciliation)
    if expected_selected_slot is None:
        frame_candidates, row_reconciliation = _reconcile_hif_unselected_detector_gap(
            context, frames, frame_candidates)
        if row_reconciliation is not None and reconciliation_events is not None:
            reconciliation_events.append(row_reconciliation)
    reference = frame_candidates[0]
    geometry_stable = bool(reference) and all(
        _same_live_candidate_geometry(reference, observed)
        for observed in frame_candidates[1:]
    )
    if (not geometry_stable and expected_selected_slot is None
            and getattr(context, 'gkh_hif_upstream_card_ui', False) is True):
        from live_decision.hif_upstream_policy import unselected_row_anchors

        anchors = [unselected_row_anchors(row, frame.shape)
            for frame, row in zip(frames, frame_candidates, strict=True)]
        geometry_stable = bool(anchors[0]) and all(row == anchors[0] for row in anchors[1:])
    select_evidence: list[dict[str, Any]] = []
    if not geometry_stable:
        return None, frame_candidates, select_evidence, False
    select_observations: list[tuple[int, ...]] = []
    for frame, observed in zip(frames, frame_candidates, strict=True):
        slots, evidence = _observe_expected_selected_live_slots(
            context,
            frame,
            observed,
            expected_selected_slot,
        )
        select_observations.append(slots)
        select_evidence.append(evidence)
    return (
        resolve_live_selected_slot_consensus(select_observations, single_frame=current_frame),
        frame_candidates,
        select_evidence,
        True,
    )


class _LiveInputNotDispatched(RuntimeError):
    """The final guard or host preparation failed before calling Maa input."""


def _live_selection_click_candidate(candidates, slot):
    """Center a first selection in the unobscured strip of an overlapping row."""
    candidate = candidates[slot]
    box = hif_selection_click_box(tuple(card.box for card in candidates), slot)
    return candidate if box == candidate.box else SimpleNamespace(box=box)


def _click_live_candidate(
    context: Context, candidate: Any, *, input_deadline: float | None = None,
    authorization_floor: float | None = None,
) -> bool:
    x, y, width, height = candidate.box
    center_x, center_y = x + width // 2, y + height // 2
    if input_deadline is not None or authorization_floor is not None:
        def before_dispatch():
            expired, observed = _validate_live_input_clock_window(
                input_deadline, authorization_floor)
            if expired:
                raise _LiveInputNotDispatched('snapshot_expired_after_input_preparation')
            return observed

        guarded_click = getattr(context, 'gkh_hif_post_live_click', None)
        if callable(guarded_click):
            job = guarded_click(center_x, center_y, before_dispatch=before_dispatch,
                input_deadline=input_deadline, authorization_floor=authorization_floor)
        else:
            before_dispatch()
            job = context.tasker.controller.post_click(center_x, center_y)
    else:
        job = context.tasker.controller.post_click(center_x, center_y)
    return bool(job.wait().succeeded)


def _eligible_family_ids(
    catalog: SkillCardFeatureCatalog,
    family_ids: tuple[str, ...],
    plan: str,
) -> tuple[str, ...]:
    return tuple(
        card_id
        for card_id in family_ids
        if card_id in catalog.entries and catalog.entries[card_id].plan in {"free", plan}
    )


def _single_frame_hand_count_outlier(rows):
    """Recapture a batch with one missing/split detection; never fill it in."""
    if len(rows) != 3 or sorted(map(len, rows)) not in ([0, 3, 3], [3, 3, 4]):
        return False
    complete = [row for row in rows if len(row) == 3]
    return _same_live_candidate_geometry(complete[0], complete[1])


def _reconcile_single_live_detector_fragment(
    frame_candidates: tuple[tuple[Any, ...], ...],
    *,
    expected_hand_count: int | None = None,
) -> tuple[tuple[tuple[Any, ...], ...], dict[str, Any] | None]:
    """Drop one uniquely proven low-score detector fragment from three frames."""

    if type(expected_hand_count) is not int or expected_hand_count != 3:
        return frame_candidates, None
    if len(frame_candidates) != 3:
        return frame_candidates, None
    counts = tuple(len(candidates) for candidates in frame_candidates)
    base_count = expected_hand_count
    if sorted(counts) != [base_count, base_count, base_count + 1]:
        return frame_candidates, None
    base_indexes = [index for index, count in enumerate(counts) if count == base_count]
    extra_index = next(
        index for index, count in enumerate(counts) if count == base_count + 1
    )
    first_base = frame_candidates[base_indexes[0]]
    second_base = frame_candidates[base_indexes[1]]
    if not _same_live_candidate_geometry(first_base, second_base):
        return frame_candidates, None

    extra = frame_candidates[extra_index]
    solutions: list[tuple[int, int, tuple[Any, ...]]] = []
    for removed_index, fragment in enumerate(extra):
        fragment_score = getattr(fragment, "detector_score", None)
        if (
            isinstance(fragment_score, bool)
            or not isinstance(fragment_score, (int, float))
            or not math.isfinite(float(fragment_score))
            or float(fragment_score) > 0.50
        ):
            continue
        fragment_x, fragment_y, fragment_width, fragment_height = fragment.box
        if fragment_width <= 0 or fragment_height <= 0:
            continue
        remaining = tuple(
            replace(candidate, slot=slot)
            for slot, candidate in enumerate(
                candidate
                for index, candidate in enumerate(extra)
                if index != removed_index
            )
        )
        if not (
            _same_live_candidate_geometry(first_base, remaining)
            and _same_live_candidate_geometry(second_base, remaining)
        ):
            continue
        for host_index, host in enumerate(extra):
            if host_index == removed_index:
                continue
            host_score = getattr(host, "detector_score", None)
            if (
                isinstance(host_score, bool)
                or not isinstance(host_score, (int, float))
                or not math.isfinite(float(host_score))
                or float(host_score) < 0.80
            ):
                continue
            host_x, host_y, host_width, host_height = host.box
            if host_width <= 0 or host_height <= 0:
                continue
            fragment_center_x = fragment_x + fragment_width / 2.0
            horizontal_overlap = max(
                0,
                min(fragment_x + fragment_width, host_x + host_width)
                - max(fragment_x, host_x),
            )
            vertical_overlap = max(
                0,
                min(fragment_y + fragment_height, host_y + host_height)
                - max(fragment_y, host_y),
            )
            if (
                fragment_width <= 0.50 * host_width
                and host_x <= fragment_center_x <= host_x + host_width
                and horizontal_overlap / fragment_width >= 0.60
                and vertical_overlap / min(fragment_height, host_height) >= 0.90
            ):
                solutions.append((removed_index, host_index, remaining))
    if len(solutions) != 1:
        return frame_candidates, None

    removed_index, host_index, remaining = solutions[0]
    reconciled = list(frame_candidates)
    reconciled[extra_index] = remaining
    fragment = extra[removed_index]
    host = extra[host_index]
    telemetry = {
        "accepted": True,
        "reason": "single_low_score_overlapping_detector_fragment",
        "original_candidate_boxes": [
            [list(candidate.box) for candidate in candidates]
            for candidates in frame_candidates
        ],
        "reconciled_candidate_boxes": [
            [list(candidate.box) for candidate in candidates]
            for candidates in reconciled
        ],
        "removed_frame_index": extra_index,
        "removed_box": list(fragment.box),
        "removed_score": float(fragment.detector_score),
        "host_box": list(host.box),
        "host_score": float(host.detector_score),
    }
    return tuple(reconciled), telemetry


def _selection_observation_preflight(
    context: Context,
    candidates: tuple[Any, ...],
    current_selected_slot: int | None,
    attempted_slot: int,
    max_snapshot_age_ms: float,
    *,
    missing_select_retries: int = 2,
    retry_single_empty_hand: bool = True,
) -> tuple[tuple[Any, ...], float, dict[str, Any]]:
    """Refresh Live anchors, hand geometry and SELECT state before one probe click."""

    try:
        captured_at = _read_strict_live_capture_clock()
    except _LiveCaptureTimingError as error:
        raise LiveSelectionError(
            error.reason,
            events=(
                {
                    "phase": "selection_observation_preflight",
                    "attempted_slot": attempted_slot,
                    "click_dispatched": False,
                    "input_may_have_been_dispatched": False,
                    "accepted": False,
                    "reason": error.reason,
                    **error.details,
                },
            ),
        ) from error
    frames = _capture_three_frames(context)
    frame_candidates: list[tuple[Any, ...]] = []
    health_votes = 0
    playable_votes = 0
    for frame in frames:
        health_detail = context.run_recognition("ProduceRecognitionHealthFlag", frame)
        playable_detail = context.run_recognition("ProduceRecognitionSkipRound", frame)
        health_votes += int(bool(health_detail and health_detail.hit))
        playable_votes += int(bool(playable_detail and playable_detail.hit))
        detail = context.run_recognition("ProduceRecognitionCards", frame)
        results = (
            list(getattr(detail, "all_results", ()) or ())
            if detail and detail.hit
            else []
        )
        observed = _normalise_observed_live_candidates(context, frame, results)
        frame_candidates.append(observed)

    geometry_stable = all(
        _same_live_candidate_geometry(candidates, observed)
        for observed in frame_candidates
    )
    consensus = None
    select_evidence: list[dict[str, Any]] = []
    if geometry_stable:
        select_observations = []
        for frame, observed in zip(frames, frame_candidates, strict=True):
            slots, evidence = _observe_expected_selected_live_slots(
                context,
                frame,
                observed,
                current_selected_slot,
            )
            select_observations.append(slots)
            select_evidence.append(evidence)
        consensus = resolve_live_selected_slot_consensus(select_observations)
    try:
        authorization_floor = _read_strict_live_capture_clock()
    except _LiveCaptureTimingError as error:
        raise LiveSelectionError(
            error.reason,
            events=(
                {
                    "phase": "selection_observation_preflight",
                    "attempted_slot": attempted_slot,
                    "click_dispatched": False,
                    "input_may_have_been_dispatched": False,
                    "accepted": False,
                    "reason": error.reason,
                    "captured_at": captured_at,
                    **error.details,
                },
            ),
        ) from error
    timing_valid = bool(
        type(max_snapshot_age_ms) in {int, float}
        and math.isfinite(float(max_snapshot_age_ms))
        and float(max_snapshot_age_ms) > 0.0
        and authorization_floor >= captured_at
    )
    capture_age_ms = (
        (authorization_floor - captured_at) * 1000.0
        if timing_valid
        else float("inf")
    )
    deadline = (
        captured_at + float(max_snapshot_age_ms) / 1000.0
        if timing_valid
        else float("nan")
    )
    timing_valid = bool(
        timing_valid
        and math.isfinite(capture_age_ms)
        and capture_age_ms >= 0.0
        and math.isfinite(deadline)
        and captured_at <= authorization_floor < deadline
    )
    health_state_stable = health_votes == 3 or (
        current_selected_slot is not None and health_votes == 0
    )
    accepted = bool(
        health_state_stable
        and playable_votes == 3
        and geometry_stable
        and consensus is not None
        and consensus.accepted
        and consensus.selected_slot == current_selected_slot
        and timing_valid
        and capture_age_ms <= max_snapshot_age_ms
    )
    if not timing_valid:
        reason = "selection_preflight_clock_invalid"
    elif playable_votes != 3:
        reason = "live_anchors_not_unanimous"
    elif not health_state_stable:
        reason = "live_health_anchor_unstable"
    elif not geometry_stable:
        reason = "live_hand_geometry_changed"
    elif consensus is None or not consensus.accepted:
        reason = "live_select_state_ambiguous"
    elif consensus.selected_slot != current_selected_slot:
        reason = "live_select_state_changed"
    elif capture_age_ms > max_snapshot_age_ms:
        reason = "selection_snapshot_expired"
    else:
        reason = "selection_preflight_passed"
    event = {
        "phase": "selection_observation_preflight",
        "attempted_slot": attempted_slot,
        "click_dispatched": False,
        "frame_ids": [frame_identifier(frame) for frame in frames],
        "health_votes": health_votes,
        "health_state_stable": health_state_stable,
        "playable_votes": playable_votes,
        "candidate_boxes": [
            [list(candidate.box) for candidate in observed]
            for observed in frame_candidates
        ],
        "select_consensus": consensus.to_dict() if consensus is not None else None,
        "select_evidence": select_evidence,
        "captured_at": captured_at,
        "capture_age_ms": capture_age_ms,
        "input_deadline": deadline,
        "authorization_floor": authorization_floor,
        "input_may_have_been_dispatched": False,
        "accepted": accepted,
        "reason": reason,
    }
    observed_selected_slots = (
        {
            slot
            for observation in consensus.observations
            for slot in observation
        }
        if consensus is not None
        else set()
    )
    retryable_missing_select = bool(
        missing_select_retries > 0
        and timing_valid
        and current_selected_slot is not None
        and health_state_stable
        and playable_votes == 3
        and geometry_stable
        and consensus is not None
        and consensus.reason == "select_frames_disagree"
        and observed_selected_slots == {current_selected_slot}
    )
    retryable_empty_hand = bool(
        retry_single_empty_hand
        and isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext)
        and current_selected_slot is None
        and len(candidates) == 5
        and timing_valid
        and health_votes == playable_votes == 3
        and sum(not row for row in frame_candidates) == 1
        and all(not row or _same_live_candidate_geometry(candidates, row)
                for row in frame_candidates)
    )
    if retryable_missing_select or retryable_empty_hand:
        if retryable_empty_hand:
            # 1611: one animation frame has no independently proven gray
            # border. Keep it rejected; one complete fresh batch must pass.
            event["retry_reason"] = "single_empty_hand_frame"
        time.sleep(SELECTION_SETTLE_SECONDS)
        retry_candidates, retry_deadline, retry_event = (
            _selection_observation_preflight(
                context,
                candidates if retryable_empty_hand else frame_candidates[-1],
                current_selected_slot,
                attempted_slot,
                max_snapshot_age_ms,
                missing_select_retries=(missing_select_retries - 1
                                        if retryable_missing_select else missing_select_retries),
                retry_single_empty_hand=False,
            )
        )
        retry_event["prior_attempts"] = [
            event,
            *list(retry_event.get("prior_attempts", [])),
        ]
        return retry_candidates, retry_deadline, retry_event
    return frame_candidates[-1], deadline, event


def _capture_selected_live_confirmation(
    context: Context,
    *,
    expected_hand_size: int,
    target_slot: int,
    retries_remaining: int = 2,
    retry_index: int = 0,
    click_dispatched: bool = True,
) -> dict[str, Any]:
    """Confirm one selection click without weakening the three-frame gate."""

    frames = _capture_three_frames(context)
    frame_candidates: list[tuple[Any, ...]] = []
    for frame in frames:
        detail = context.run_recognition("ProduceRecognitionCards", frame)
        results = (
            list(getattr(detail, "all_results", ()) or ())
            if detail and detail.hit
            else []
        )
        frame_candidates.append(
            _normalise_observed_live_candidates(context, frame, results)
        )
    geometry_stable = bool(
        frame_candidates
        and len(frame_candidates[0]) == expected_hand_size
        and all(
            _same_live_candidate_geometry(frame_candidates[0], observed)
            for observed in frame_candidates[1:]
        )
    )
    consensus = None
    select_evidence: list[dict[str, Any]] = []
    if geometry_stable:
        select_observations = []
        for frame, observed in zip(frames, frame_candidates, strict=True):
            slots, evidence = _observe_expected_selected_live_slots(
                context,
                frame,
                observed,
                target_slot,
            )
            select_observations.append(slots)
            select_evidence.append(evidence)
        consensus = resolve_live_selected_slot_consensus(select_observations)
    accepted = bool(
        geometry_stable
        and consensus is not None
        and consensus.accepted
        and consensus.selected_slot == target_slot
    )
    event = {
        "phase": (
            "selection_observation"
            if retry_index == 0
            else "selection_observation_retry"
        ),
        "retry_index": retry_index,
        "attempted_slot": target_slot,
        "click_dispatched": click_dispatched,
        "frame_ids": [frame_identifier(frame) for frame in frames],
        "candidate_boxes": [
            [list(candidate.box) for candidate in observed]
            for observed in frame_candidates
        ],
        "geometry_stable": geometry_stable,
        "select_consensus": consensus.to_dict() if consensus is not None else None,
        "select_evidence": select_evidence,
        "accepted": accepted,
    }
    observed_slots = (
        {
            slot
            for observation in consensus.observations
            for slot in observation
        }
        if consensus is not None
        else set()
    )
    retryable = bool(
        not accepted
        and retries_remaining > 0
        and (
            not geometry_stable
            or (
                consensus is not None
                and observed_slots.issubset({target_slot})
            )
        )
    )
    if retryable:
        time.sleep(SELECTION_SETTLE_SECONDS)
        retry = _capture_selected_live_confirmation(
            context,
            expected_hand_size=expected_hand_size,
            target_slot=target_slot,
            retries_remaining=retries_remaining - 1,
            retry_index=retry_index + 1,
            click_dispatched=False,
        )
        retry["events"] = [event, *retry["events"]]
        return retry
    return {
        "accepted": accepted,
        "frames": frames,
        "frame_candidates": frame_candidates,
        "consensus": consensus,
        "events": [event],
    }


def _live_card_art_profile(context, candidates, selected_slot) -> dict[str, Any]:
    if isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext):
        if len(candidates) in (4, 5):
            profile = "hif-four-card-art-v1" if len(candidates) == 4 else "hif-five-card-art-v2"
            result = {"layout_profile": profile, "selected_slot": selected_slot}
            prepared = getattr(context, "gkh_hif_card_art", None)
            if isinstance(prepared, FrozenMaskedArt):
                result["prepared_art"] = prepared
            return result
        result = {'refine_hif_title_anchor': True} if selected_slot is None and len(candidates) in (2,3) else {}
        if 1 <= len(candidates) <= 3:
            prepared = getattr(context, 'gkh_hif_card_art', None)
            if isinstance(prepared, FrozenMaskedArt):
                result.update(prepared_art=prepared, hif_full_art_mask=True)
        return result
    if len(candidates) == 3:
        return {"legacy_art_boxes": LIVE_CARD_ART_BOXES}
    return {}


def _retained_hif_hand_ids(context, families):
    """Reuse a complete verified hand only while its fresh families still match."""
    from live_decision.inventory import match_inventory_hand

    bundle = getattr(context, 'gkh_hif_opening_state', None)
    event = getattr(context, 'gkh_hif_committed_event', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    if event is None and getattr(context, 'gkh_hif_extended_opening_identity', None) is not None:
        from live_decision.extended_opening import initial_bound_hand_ids
        try:
            return initial_bound_hand_ids(context, families, SkillCardFeatureCatalog.load(CATALOG_PATH))
        except (ValueError, KeyError, AttributeError) as error:
            raise LiveSelectionError('retained HIF opening hand: '+str(error)) from error
    rule_maintained = isinstance(event,dict) and event.get('verification_scope') == 'rule_maintained_information_boundary_v1'
    opening_maintained = False
    if (bundle is not None and isinstance(event, dict) and event.get('event') == 'live_opening_state'
            and event.get('extended_hand_identity') is not None and run is not None):
        from live_decision.extended_opening import verify_extended_opening_identity
        from live_decision.high_score.maintenance import complete_action_baseline

        if complete_action_baseline(bundle, event):
            ids = tuple(bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id
                        for ref in bundle.state.deck.hand)
            verified = verify_extended_opening_identity(event['extended_hand_identity'], run=run,
                core=event['core'], inventory_rows=[dict(zone='手札', card_id=cid) for cid in ids],
                catalog=SkillCardFeatureCatalog.load(CATALOG_PATH))
            opening_maintained = verified == ids
    if (bundle is None or not isinstance(event, dict) or run is None
            or event.get('accepted') is not True
            or (not opening_maintained and (not event.get('checks')
                or not all(v is True for v in event['checks'].values())))
            or event.get('run_id') != bundle.run_id or event.get('revision') != bundle.revision
            or (run.run_id, run.revision) != (bundle.run_id, bundle.revision)
            or len(set(event.get('core', {}).get('frame_ids', ()))) != 3
            or (not rule_maintained
                and tuple(tuple(f) for f in event['core'].get('families', ())) != tuple(families))):
        raise LiveSelectionError('retained HIF hand lacks matching complete evidence')
    if rule_maintained:
        # A derived exact card ID is not an old visual-family observation.
        # Validate the exact rule-bound order, then compare each fresh family
        # to its known slot. Normal/+ copies need not be reassigned as an
        # unordered inventory; target/title/input guards remain fresh.
        from live_decision.high_score.hif_rule_progress import _baseline
        try:
            _baseline(bundle,event.get('rule_progress',{}))
        except ValueError as error:
            raise LiveSelectionError('retained HIF rule hand is unbound') from error
    ids = tuple(bundle.kernel.cards[r.card_id].canonical_card_id or r.card_id for r in bundle.state.deck.hand)
    matches = (len(families) == len(ids) and all(cid in family for cid, family in zip(ids, families, strict=True))
        if rule_maintained or opening_maintained else match_inventory_hand(families, ids) == ids)
    if not matches:
        raise LiveSelectionError('retained HIF hand order changed')
    return ids


def _hif_selected_detail_identity(context, images, family_ids, catalog, plan):
    """Read complete popup titles; callers have already confirmed SELECT."""
    from copy import deepcopy

    import numpy as np
    from live_decision.upgrade import normalize_upgrade_title
    from live_decision.hif_single_plus import normalize_hif_title, retained_title_row_valid
    from live_decision.hif_skill_support import _text

    evidence = []
    resolved = []
    title_ocr = []
    for image in images:
        roi = locate_hif_selected_detail_title(image)
        item = {"frame_id": frame_identifier(image), "roi": roi, "card_id": None}
        if roi is not None:
            x, y, w, h = roi
            node = "Task275HifSelectedDetailTitle"
            crop = image[y:y+h, x:x+w].copy()
            texts = None
            # Only this fixed OCR call shares computation inside this one
            # three-frame read. Every frame still locates its own full panel,
            # supplies fresh pixels and independently resolves its title.
            for previous_roi, previous_pixels, previous_rows in title_ocr:
                if (roi == previous_roi and crop.shape == previous_pixels.shape
                        and crop.dtype == previous_pixels.dtype
                        and np.array_equal(crop, previous_pixels)):
                    texts = deepcopy(previous_rows)
                    break
            if texts is None:
                hit = context.run_recognition(node, crop,
                    pipeline_override={node: {"recognition": "OCR", "expected": ".+"}})
                raw = hit.raw_detail if hit else {}
                texts = deepcopy(raw.get("all", [])) if isinstance(raw, dict) else []
                if isinstance(texts, list) and len(texts) == 1 and .85 <= texts[0].get('score', 0) < .95:
                    from live_decision.hif_single_plus import read_selected_title_frame
                    try:
                        title = read_selected_title_frame(context,image,frame_id=item['frame_id'])
                    except ValueError:
                        title = None
                    if title is not None and any(row.get('acquisition_mode') in (
                            'complete_title_horizontal_trim','complete_name_and_observed_plus_suffix',
                            'retained_complete_hand_selected_title')
                            for row in title['title_rows']):
                        item['original_ocr'] = deepcopy(texts)
                        texts = [dict(text=row['text'],score=row['confidence'],
                            box=[row['box'][0]-x,row['box'][1]-y,*row['box'][2:]],
                            acquisition_mode=row.get('acquisition_mode'),source_roi=row.get('source_roi'),
                            **({key: row[key] for key in ('base_title_rows','suffix_components')}
                               if row.get('acquisition_mode') == 'complete_name_and_observed_plus_suffix' else {}),
                            **({key: row[key] for key in ('retained_identity','text_accepted')}
                               if row.get('acquisition_mode') == 'retained_complete_hand_selected_title' else {}))
                            for row in title['title_rows']]
                if isinstance(texts, list):
                    title_ocr.append((roi, crop.copy(), deepcopy(texts)))
            item["ocr"] = texts
            matches = []
            for row in texts:
                box = row.get("box", ())
                retained = row.get('retained_identity', {})
                retained_ok = bool(len(box) == 4 and retained_title_row_valid(
                    dict(row,confidence=row.get('score',0),box=[box[0]+x,box[1]+y,*box[2:]]),
                    getattr(context,'gkh_hif_opening_state',None),retained.get('binding',{}).get('slot')))
                if (len(box) != 4 or row.get("score", 0) < 0.95 and not retained_ok
                        or box[0] < 0 or box[1] < 0
                        or box[0] + box[2] > w or box[1] + box[3] > h):
                    continue
                # D-047 folds additional enhancement tiers for identity only.
                # Missing plus signs and partial names still do not match;
                # the separate support receipt proves the actual tier/effects.
                title = normalize_hif_title(str(row.get("text", "")))
                matches.extend(card_id for card_id in _eligible_family_ids(catalog, family_ids, plan)
                               if normalize_upgrade_title(title) == normalize_upgrade_title(
                                   _text(catalog.entries[card_id].name)))
            if len(matches) == 1:
                item["card_id"] = matches[0]
        evidence.append(item)
        resolved.append(item["card_id"])
    accepted = len(resolved) == 3 and resolved[0] is not None and len(set(resolved)) == 1
    return {"accepted": accepted, "card_id": resolved[0] if accepted else None,
            "observations": evidence}


def _hif_selected_upgrade_consensus(context, images, family_ids, catalog, plan):
    detail = _hif_selected_detail_identity(context, images, family_ids, catalog, plan)
    entry = catalog.entries.get(detail["card_id"])
    return LiveUpgradeConsensus(
        accepted=bool(detail["accepted"]),
        upgraded=entry.upgraded if entry else None,
        plus_votes=3 if entry and entry.upgraded else 0,
        normal_votes=3 if entry and not entry.upgraded else 0,
        reason="hif_complete_title_unanimous" if entry else "hif_complete_title_unresolved",
        observations=(),
    ), detail


def _bound_selected_reset_scope(context):
    """An audited, unsubmitted selection may be cleared before resolving its overlap."""
    from live_decision.hif_bound_title import binding

    journal = getattr(context, 'gkh_hif_gray_selected_source', None)
    complete = getattr(context, 'gkh_hif_committed_event', None)
    if not isinstance(journal, dict) or not isinstance(complete, dict):
        raise LiveSelectionError('bound selection reset lacks its verified source')
    a = journal.get('refresh_a', {})
    action = a.get('proposal', {}).get('candidate_traces', {}).get('high_score', {}).get('action', {})
    slot = action.get('source_slot')
    core = complete.get('core', {})
    families = tuple(tuple(f) for f in core.get('families', ()))
    scope = _inventory_inspected_return_scope(context, core)
    if (type(slot) is not int or not 0 <= slot < len(families)-1
            or len(families) not in (4, 5) or binding(context, slot) is None
            or binding(context, slot+1) is None or scope is None or scope['slot'] != slot+1
            or journal.get('ordinary_safe_preview') is not True
            or journal.get('selection_ready') is not True
            or type(journal.get('selection_count')) is not int or journal['selection_count'] != 1
            or journal.get('business_submit_count') != 0 or a.get('accepted') is not True
            or a.get('expected_selected_slot') is not None
            or a.get('unselected_geometry_authorized') is not True):
        raise LiveSelectionError('bound selection reset lacks its independent original hand')
    identities = a.get('identity_revalidation', ())
    if (len(identities) != len(families) or any(
            item.get('slot') != i or item.get('accepted') is not True
            or item.get('standard_accepted') is not True or item.get('carried') is not False
            or tuple(item.get('fresh_family', ())) != families[i]
            or tuple(item.get('expected_family', ())) != families[i]
            for i, item in enumerate(identities))):
        raise LiveSelectionError('bound selection reset original identity differs')
    return dict(selected_slot=slot, families=families, core=core, return_scope=scope)


def _cancel_hif_live_selection(context, params, *, inspect_slot=None, bound_selected_reset=False,
                               pending_selected_slot=None):
    """Cancel one verified HIF selection, without selecting or playing a card."""
    from utils.maa_input import verify_maa_input_endpoints

    events = []
    try:
        maximum_age_ms = params["max_snapshot_age_ms"]
        if (not isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext)
                or type(maximum_age_ms) not in (int, float)
                or not math.isfinite(maximum_age_ms) or not 0 < maximum_age_ms <= 1000):
            raise LiveSelectionError("selection reset requires calibrated HIF and bounded age")
        preview_only = pending_selected_slot is not None
        if preview_only and (type(pending_selected_slot) is not int or not 0 <= pending_selected_slot < 5
                             or bound_selected_reset or inspect_slot is not None):
            raise LiveSelectionError('pending preview reset requires one original slot')
        # A neutral stage click needs a current selected panel, not the meaning
        # of every card behind it. Identity/resources are read after it closes.
        recognizer = None
        if not preview_only:
            recognizer = EmbeddingCardRecognizer.load(MODEL_ROOT)
            context.gkh_hif_card_art = FrozenMaskedArt.load(recognizer, MODEL_ROOT / "hif")
        if type(bound_selected_reset) is not bool or bound_selected_reset and inspect_slot is not None:
            raise LiveSelectionError('bound selection reset cannot create an inspection reference')
        bound_reset = _bound_selected_reset_scope(context) if bound_selected_reset else None
        reset_catalog = SkillCardFeatureCatalog.load(CATALOG_PATH) if bound_reset is not None else None

        inspected = None
        after_clear = False
        latest_frames = ()
        empty_hand_retry_geometry = None

        def checkpoint(frames, expected_slot):
            nonlocal latest_frames, empty_hand_retry_geometry
            latest_frames = frames
            consensus, rows, evidence, stable = _observe_live_selected_slot_consensus(
                context, frames, expected_selected_slot=expected_slot)
            if preview_only and not stable and len(rows) == 3:
                complete = [row for row in rows if row]
                if len(complete) == 2 and _same_live_candidate_geometry(complete[0], complete[1]):
                    raise LiveSelectionError('pending preview has one missing hand frame', events=(dict(
                        reason='single_frame_hand_count_outlier',
                        frame_ids=[frame_identifier(frame) for frame in frames],
                        candidate_boxes=[[list(card.box) for card in row] for row in rows]),))
            if (not stable and after_clear and expected_slot is None
                    and inspected is not None and inspected.get('independent_ready') is True
                    and len(inspected['selected']['families']) == 5
                    and len(rows) == 3 and sorted(map(len, rows)) == [0, 5, 5]):
                complete = [(frame, row) for frame, row in zip(frames, rows, strict=True) if row]
                if not _same_live_candidate_geometry(complete[0][1], complete[1][1]):
                    raise LiveSelectionError('inspection return complete hand rows differ')
                complete_select = []
                for frame, row in complete:
                    slots, proof = _observe_expected_selected_live_slots(context, frame, row, None)
                    if (slots or not proof.get('geometry_available') or proof.get('geometry_ambiguous')
                            or proof.get('geometry_raised_slots') or proof.get('ocr_slots')
                            or proof.get('resolved_slots')):
                        raise LiveSelectionError('inspection return complete row is not unselected')
                    complete_select.append(proof)
                if empty_hand_retry_geometry is None:
                    empty_hand_retry_geometry = complete[0][1]
                raise LiveSelectionError('inspection return has one empty hand frame', events=(dict(
                    reason='inspected_single_empty_hand_frame',
                    frame_ids=[frame_identifier(frame) for frame in frames],
                    candidate_boxes=[[list(card.box) for card in row] for row in rows],
                    complete_select_evidence=complete_select),))
            if not stable and _single_frame_hand_count_outlier(rows):
                raise LiveSelectionError('selection reset has one detector split frame', events=(dict(
                    reason='single_frame_hand_count_outlier', frame_ids=[frame_identifier(frame) for frame in frames],
                    candidate_boxes=[[list(card.box) for card in row] for row in rows]),))
            if not stable or consensus is None or not consensus.accepted:
                raise LiveSelectionError("selection reset requires stable SELECT and geometry")
            if preview_only:
                if consensus.selected_slot is not None:
                    if consensus.selected_slot != pending_selected_slot:
                        raise LiveSelectionError('pending preview reset target changed')
                    for frame in frames:
                        title = locate_hif_selected_detail_title(frame)
                        if title is None or title[1] < 360:
                            raise LiveSelectionError('selection reset stage region is covered')
                return dict(selected_slot=consensus.selected_slot,
                    frame_ids=[frame_identifier(frame) for frame in frames],
                    select_evidence=evidence, candidates=rows[-1])
            if (after_clear and empty_hand_retry_geometry is not None
                    and any(not _same_live_candidate_geometry(empty_hand_retry_geometry, row) for row in rows)):
                raise LiveSelectionError('inspection return changed after empty hand observation')
            if bound_reset is not None and consensus.selected_slot != bound_reset['selected_slot']:
                raise LiveSelectionError('bound selection reset target changed')
            if bound_reset is not None:
                original = context.gkh_hif_gray_selected_source['refresh_a'].get('candidate_boxes', ())
                if len(original) != 3 or any(len(row) != len(bound_reset['families']) for row in original):
                    raise LiveSelectionError('bound selection reset original geometry is incomplete')
                selected = bound_reset['selected_slot']
                for row in rows:
                    lowered = [replace(c, box=(c.box[0], original[-1][selected][1], *c.box[2:]))
                        if c.slot == selected else c for c in row]
                    if not _same_live_candidate_geometry(
                            tuple(SimpleNamespace(box=tuple(box)) for box in original[-1]), tuple(lowered)):
                        raise LiveSelectionError('bound selection reset physical hand changed')
            families, turns, identity_observations = [], [], []
            selected_title_pending = False
            original_rows = []
            weak_returned_target = False
            inspect_return = (after_clear and inspected is not None
                and inspected.get('independent_ready') is True and consensus.selected_slot is None)
            for frame, row in zip(frames, rows, strict=True):
                if consensus.selected_slot is not None:
                    title = locate_hif_selected_detail_title(frame)
                    # The panel border is ten pixels above its title region.
                    # Keep ten more pixels clear below the stage target's 340
                    # bottom edge; a long description may raise the panel.
                    if title is None or title[1] < 360:
                        raise LiveSelectionError("selection reset stage region is covered")
                predictions = classify_live_card_views(recognizer, frame, row,
                    frame_id=frame_identifier(frame), plan=_infer_plan(context),
                    acceptance_threshold=params["debug_identity_threshold"],
                    min_margin=params["debug_identity_min_margin"],
                    **_live_card_art_profile(context, row, consensus.selected_slot))
                original_predictions = [p.to_dict() for p in predictions]
                original_rows.append(predictions)
                title_families, title_proofs, neighbor_proofs = {}, [], []
                if inspect_slot is not None:
                    if consensus.selected_slot == inspect_slot:
                        from live_decision.hif_inspected_art import SELECTED_TITLE_SCOPE
                        from live_decision.hif_inspection_neighbor import corroborate

                        proof = corroborate(context, frame, row, predictions,
                            selected_slot=inspect_slot, evidence=neighbor_proofs)
                        if proof is not None:
                            title_families[proof['slot']] = tuple(proof['family'])
                        target = predictions[inspect_slot]
                        if (not target.accepted or inspected is not None
                                and inspected.get('title_scope') == SELECTED_TITLE_SCOPE):
                            # This is a provisional family only. The complete
                            # three-frame title/dual-candidate proof below must
                            # pass before this checkpoint can authorize navigation.
                            if not target.top_k_card_ids:
                                raise LiveSelectionError('inspection selected family has no candidate')
                            title_families[inspect_slot] = tuple(target.top_k_card_ids[0])
                            selected_title_pending = True
                    for candidate, prediction in zip(row, predictions, strict=True):
                        if (prediction.accepted or candidate.slot == consensus.selected_slot
                                or candidate.slot in title_families):
                            continue
                        proof = _inventory_full_title_family(context, frame, candidate, prediction,
                            minimum_margin=params['debug_identity_min_margin'], allow_low_margin_title=True,
                            unselected_candidates=row)
                        if proof is not None:
                            title_families[candidate.slot] = tuple(proof['family'])
                            title_proofs.append(proof)
                if after_clear and inspected is not None and consensus.selected_slot is None:
                    predictions = tuple(replace(p, accepted=True)
                        if (p.slot == inspect_slot
                            and (p.reason == 'below_margin_threshold' and p.confidence >= .85
                                or inspected.get('independent_ready') is True
                                and p.reason in ('below_acceptance_threshold', 'below_margin_threshold'))
                            and p.recognizer == 'card_art_embedding_top5_hif_masked_v1'
                            and math.isfinite(p.confidence) and 0 <= p.confidence <= 1
                            and p.top_k_card_ids
                            and tuple(p.top_k_card_ids[0]) == tuple(inspected['family']))
                        else p for p in predictions)
                deferred_slots = set()
                if bound_reset is not None:
                    from live_decision.hif_inspected_art import accepted_inspection_full_view

                    selected = bound_reset['selected_slot']
                    if (len(predictions) != len(bound_reset['families'])
                            or any(p.slot != i or p.frame_id != frame_identifier(frame)
                                or tuple(p.box) != tuple(row[i].box) or not p.top_k_card_ids
                                for i, p in enumerate(predictions))):
                        raise LiveSelectionError('bound selection reset hand geometry changed')
                    for p in predictions:
                        expected = bound_reset['families'][p.slot]
                        if (p.slot == selected+1 and not p.accepted
                                and p.reason in ('below_acceptance_threshold', 'below_margin_threshold')
                                and p.recognizer == 'card_art_embedding_top5_hif_masked_v1'
                                and math.isfinite(p.confidence) and 0 <= p.confidence <= 1
                                and math.isfinite(p.margin) and 0 <= p.margin <= 1):
                            deferred_slots.add(p.slot)
                        elif not p.accepted or tuple(p.top_k_card_ids[0]) != expected:
                            raise LiveSelectionError('bound selection reset another identity is unresolved or changed')
                    if not accepted_inspection_full_view(predictions[selected].to_dict(),
                            frame_id=frame_identifier(frame), family=bound_reset['families'][selected]):
                        raise LiveSelectionError('bound selection reset target lacks its complete strong view')
                if inspect_return:
                    if len(predictions) != len(inspected['selected']['families']):
                        raise LiveSelectionError('selection reset did not preserve hand and turn')
                    for slot,p in enumerate(predictions):
                        expected_family = tuple(inspected['selected']['families'][slot])
                        if p.slot != slot or not p.top_k_card_ids:
                            raise LiveSelectionError('selection reset hand identity unresolved')
                        raw_family = tuple(p.top_k_card_ids[0])
                        family = title_families.get(slot, raw_family)
                        if (slot == inspect_slot and not p.accepted and raw_family != expected_family
                                and p.reason in ('below_acceptance_threshold', 'below_margin_threshold')
                                and p.recognizer == 'card_art_embedding_top5_hif_masked_v1'
                                and p.frame_id == frame_identifier(frame)
                                and math.isfinite(p.confidence) and 0 <= p.confidence <= 1
                                and math.isfinite(p.margin) and 0 <= p.margin <= 1
                                and (slot not in title_families or family == expected_family)):
                            deferred_slots.add(slot)
                            weak_returned_target = True
                        elif family != expected_family:
                            raise LiveSelectionError('selection reset did not preserve hand and turn')
                if len(predictions) != len(row) or any(
                        (not p.accepted and p.slot not in title_families and p.slot not in deferred_slots)
                        or not p.top_k_card_ids for p in predictions):
                    weak = [p for p in predictions if not p.accepted]
                    if (not after_clear and inspect_slot is None and bound_reset is None
                            and consensus.selected_slot is not None and len(row) in (4, 5)
                            and len(predictions) == len(row) and len(weak) == 1
                            and all(p.slot == i and p.top_k_card_ids for i, p in enumerate(predictions))
                            and weak[0].reason in ('below_acceptance_threshold', 'below_margin_threshold')
                            and weak[0].recognizer == 'card_art_embedding_top5_hif_masked_v1'
                            and math.isfinite(weak[0].confidence) and 0 <= weak[0].confidence <= 1
                            and math.isfinite(weak[0].margin) and 0 <= weak[0].margin <= 1):
                        # Observation only: the failed batch authorizes no input.
                        # The existing two-attempt budget must obtain an entirely
                        # new, fully accepted batch, followed by fresh preflight.
                        raise LiveSelectionError('selection reset hand identity unresolved', events=(dict(
                            reason='selection_reset_single_weak_identity',
                            frame_ids=[frame_identifier(f) for f in frames],
                            failed_frame_id=frame_identifier(frame), predictions=original_predictions,
                            input_click_count=0),))
                    raise LiveSelectionError("selection reset hand identity unresolved")
                families.append(tuple(None if p.slot in deferred_slots
                    else title_families.get(p.slot, tuple(p.top_k_card_ids[0])) for p in predictions))
                identity_observations.append(dict(frame_id=frame_identifier(frame),
                    predictions=original_predictions, title_proofs=title_proofs,
                    **({'neighbor_proofs': neighbor_proofs} if neighbor_proofs else {})))
                turn = _ocr_field(context, frame, "turns_remaining", *LIVE_LAYOUT_V1["turns_remaining"])
                if not turn.get("accepted") or turn.get("confidence", 0) < .95:
                    raise LiveSelectionError("selection reset turn unresolved")
                turns.append(turn["value"])
            if weak_returned_target:
                if (len(frames) != 3 or len(identity_observations) != 3
                        or len({frame_identifier(frame) for frame in frames}) != 3
                        or any(turn != inspected['selected']['turns_remaining'] for turn in turns)
                        or not fixed_left_half_arts(dict(candidates=rows[-1],
                            identity_observations=identity_observations))):
                    raise LiveSelectionError('selection reset did not preserve hand and turn')
                # This complete batch remains unresolved. It only permits a
                # fresh batch within the existing shared observation budget.
                raise LiveSelectionError('inspection return target identity is temporarily unresolved',
                    events=(dict(reason='inspected_target_weak_identity', inspect_slot=inspect_slot,
                        selected_slot=None, frame_ids=[frame_identifier(f) for f in frames],
                        candidate_boxes=[[list(c.box) for c in row] for row in rows],
                        turns_remaining=turns, identity_observations=identity_observations,
                        input_click_count=0),))
            if bound_reset is not None:
                if len(set(frame_identifier(f) for f in frames)) != 3 or any(
                        turn != bound_reset['core']['fields'][0] for turn in turns):
                    raise LiveSelectionError('bound selection reset frame or turn changed')
                # Keep the overlap explicitly unknown. No old family is inserted
                # into this navigation-only observation or a business certificate.
                families = [tuple(None if any(f[i] is None for f in families) else families[0][i]
                    for i in range(len(bound_reset['families'])))]*3
                resource = _observe_recovered_selected_baseline(context, frames,
                    complete_event=context.gkh_hif_committed_event,
                    catalog=reset_catalog, plan=_infer_plan(context),
                    selection_observation=(consensus, rows, evidence, stable),
                    cost_context=_SelectedCostOcrContext(context))
                if resource.get('accepted') is not True:
                    raise LiveSelectionError('bound selection reset resource preview unresolved')
            if any(f != families[0] for f in families) or len(set(turns)) != 1:
                raise LiveSelectionError("selection reset state changed across frames")
            selected_title = None
            if selected_title_pending:
                from live_decision.hif_inspected_art import selected_title_proof

                inspection_catalog = SkillCardFeatureCatalog.load(CATALOG_PATH)
                detail = _hif_selected_detail_identity(context, frames, families[0][inspect_slot],
                    inspection_catalog, _infer_plan(context))
                selected_title = selected_title_proof(frames, rows, original_rows, detail,
                    recognizer, inspection_catalog, selected_slot=inspect_slot, plan=_infer_plan(context),
                    acceptance_threshold=params['debug_identity_threshold'], min_margin=params['debug_identity_min_margin'],
                    prepared_art=context.gkh_hif_card_art)
                if selected_title is None:
                    raise LiveSelectionError('inspection selected complete title/candidates disagree', events=[detail])
            return {"selected_slot": consensus.selected_slot, "families": families[0],
                    "turns_remaining": turns[0], "frame_ids": [frame_identifier(f) for f in frames],
                    "select_evidence": evidence, "candidates": rows[-1],
                    **({'identity_observations': identity_observations}
                       if inspect_slot is not None or bound_reset is not None else {}),
                    **({'selected_title_proof': selected_title} if selected_title is not None else {}),
                    **({'resource_baseline': resource} if bound_reset is not None else {})}

        def fixed_left_half_arts(observed):
            from live_decision.hif_inspected_art import LEFT_HALF_PROFILE, visible_art

            try:
                return [visible_art(frame, tuple(replace(c, box=tuple(p['box']))
                    for c,p in zip(observed['candidates'], observation['predictions'], strict=True)), inspect_slot,
                    visibility_profile=LEFT_HALF_PROFILE)
                    for frame,observation in zip(latest_frames, observed['identity_observations'], strict=True)]
            except ValueError:
                return []

        def bounded_checkpoint(expected_slot):
            after_inspection = (after_clear and type(inspect_slot) is int
                and inspected is not None and inspected.get('independent_ready') is True)
            maximum_attempts = 8 if after_inspection else 2
            observation_started = time.perf_counter()
            observation_deadline = observation_started + 10. if after_inspection else None
            empty_hand_retry_used = False

            def observation_timing(attempt, checked_at=None):
                return dict(observation_stage='after_clear', observation_attempt=attempt+1,
                    elapsed_ms=((time.perf_counter() if checked_at is None else checked_at)-observation_started)*1000,
                    max_observation_attempts=8, observation_budget_ms=10000)

            def check_observation_deadline(attempt):
                if observation_deadline is None:
                    return None
                checked_at = time.perf_counter()
                if checked_at >= observation_deadline:
                    events.append(dict(phase='selection_reset_observation_stop',
                        reason='inspection_return_observation_budget_expired', input_click_count=0,
                        **observation_timing(attempt, checked_at)))
                    raise LiveSelectionError('inspection return observation budget expired')
                return observation_timing(attempt, checked_at)

            for attempt in range(maximum_attempts):
                check_observation_deadline(attempt)
                captured_at = time.perf_counter()
                try:
                    observed = checkpoint(_capture_three_frames(context), expected_slot)
                    if after_clear and inspected is not None and inspected.get('independent_ready') is True:
                        if observed['selected_slot'] is not None or any(
                                observed[k] != inspected['selected'][k] for k in ('families', 'turns_remaining')):
                            raise LiveSelectionError('selection reset did not preserve hand and turn')
                        half_arts = fixed_left_half_arts(observed)
                        if half_arts and any(art != half_arts[0] for art in half_arts[1:]):
                            raise LiveSelectionError('inspection reset returned artwork is not pixel stable',
                                events=(dict(reason='inspected_left_half_pixel_unstable',
                                    frame_ids=observed['frame_ids'], input_click_count=0),))
                    timing = check_observation_deadline(attempt)
                    if after_inspection:
                        observed['observation_budget'] = timing
                    return observed, captured_at
                except LiveSelectionError as error:
                    empty_hand_retry = (len(error.events) == 1
                        and error.events[0].get('reason') == 'inspected_single_empty_hand_frame')
                    if (attempt+1 >= maximum_attempts or len(error.events) != 1
                            or empty_hand_retry and empty_hand_retry_used
                            or error.events[0].get("reason") not in {
                                "exact_select_low_confidence", "single_frame_hand_count_outlier",
                                "selection_reset_single_weak_identity",
                                "inspected_left_half_pixel_unstable", "inspected_target_weak_identity",
                                "inspected_single_empty_hand_frame"}):
                        if len(error.events) == 1 and error.events[0].get('reason') == 'inspected_target_weak_identity':
                            events.append(dict(phase='selection_reset_observation_stop',
                                reason=error.events[0]['reason'], input_click_count=0, evidence=error.events,
                                **(observation_timing(attempt) if after_inspection else {})))
                        raise
                    empty_hand_retry_used |= empty_hand_retry
                    timing = check_observation_deadline(attempt)
                    events.append({"phase": "selection_reset_observation_retry",
                                   "reason": error.events[0]['reason'], "input_click_count": 0,
                                   "evidence": error.events,
                                   **(dict(timing,
                                       observation_frame_ids=[frame_identifier(f) for f in latest_frames])
                                      if after_inspection else {})})
            raise AssertionError("bounded observation exhausted")

        before, _ = bounded_checkpoint(None)
        if inspect_slot is not None:
            if type(inspect_slot) is not int or before['selected_slot'] != inspect_slot:
                raise LiveSelectionError('inspection reset requires the exact current selected slot')
            catalog = SkillCardFeatureCatalog.load(CATALOG_PATH)
            detail = (before['selected_title_proof']['detail'] if 'selected_title_proof' in before
                else _hif_selected_detail_identity(context, latest_frames,
                    before['families'][inspect_slot], catalog, _infer_plan(context)))
            if not detail['accepted']:
                raise LiveSelectionError('inspection reset complete title is unresolved', events=[detail])
            inspected = dict(detail=detail, family=before['families'][inspect_slot],
                selected_frame_ids=before['frame_ids'], selected={k:v for k,v in before.items() if k != 'candidates'})
            if 'selected_title_proof' in before:
                inspected['title_scope'] = before['selected_title_proof']['scope']
        if before["selected_slot"] is None:
            events.append({"phase": "selection_reset", "accepted": True,
                           "reason": "already_unselected", "input_click_count": 0})
            return events
        fresh, captured_at = bounded_checkpoint(before["selected_slot"])
        comparison = ('selected_slot',) if preview_only else ('selected_slot', 'turns_remaining') if bound_reset is not None else (
            'selected_slot', 'families', 'turns_remaining')
        if any(fresh[k] != before[k] for k in comparison):
            raise LiveSelectionError("selection reset preinput state changed")
        if inspected is not None:
            # Only an originally accepted full view plus the independently
            # read title can establish a lower-scoring returned-art reference.
            from live_decision.hif_inspected_art import SELECTED_TITLE_SCOPE, accepted_inspection_full_view

            strong_full_view = all(
                accepted_inspection_full_view(p,
                    frame_id=observation['frame_id'], family=inspected['family'])
                for checkpoint in (before, fresh) for observation in checkpoint['identity_observations']
                for p in [observation['predictions'][inspect_slot]])
            independent_title = all(checkpoint.get('selected_title_proof', {}).get('scope') == SELECTED_TITLE_SCOPE
                                    for checkpoint in (before, fresh))
            if bool(inspected.get('title_scope')) != bool(fresh.get('selected_title_proof')):
                raise LiveSelectionError('inspection selected identity proof changed before cancellation')
            inspected['independent_ready'] = strong_full_view or independent_title
        guarded_input = callable(getattr(context, 'gkh_hif_post_live_click', None))
        endpoints = (None if guarded_input else verify_maa_input_endpoints(
            context.tasker.controller.info,
            expected_client_size=getattr(context, "gkh_hif_client_size", (720, 1280))))
        authorization_floor = time.perf_counter()
        age_ms = (authorization_floor - captured_at) * 1000
        if not 0 <= age_ms <= maximum_age_ms:
            events.append(dict(phase='selection_reset_expired', accepted=False,
                reason='selection reset snapshot expired', capture_age_ms=age_ms,
                max_snapshot_age_ms=maximum_age_ms, frame_ids=fresh['frame_ids'], input_click_count=0))
            raise LiveSelectionError("selection reset snapshot expired")
        event = {"phase": "selection_reset", "accepted": False, "input_click_count": 1,
                 "target": [340, 300, 40, 40], "capture_age_ms": age_ms,
                 "before": {k:v for k,v in fresh.items() if k != 'candidates'}}
        if preview_only:
            event['navigation_scope'] = 'pending_preview_only'
        if guarded_input:
            event['endpoint_evidence_source'] = 'native_input_dispatches'
        else:
            event['endpoints'] = endpoints
        events.append(event)
        # Calibrated centre-stage area, above the independently located panel,
        # below the score banner, away from status/item/card and bottom controls.
        try:
            cleared = _click_live_candidate(context, SimpleNamespace(box=event['target']),
                input_deadline=captured_at+maximum_age_ms/1000., authorization_floor=authorization_floor)
        except _LiveInputNotDispatched:
            event['input_click_count'] = 0
            event['input_may_have_been_dispatched'] = False
            raise
        if not cleared:
            raise LiveSelectionError("selection reset input failed")
        time.sleep(SELECTION_SETTLE_SECONDS)
        after_clear = True
        if bound_reset is not None:
            _, core = _observe_inventory_return(context, recognizer, bound_reset['core'], event, events,
                dict(identity_threshold=params['debug_identity_threshold'],
                     identity_min_margin=params['debug_identity_min_margin']),
                validated_scope=bound_reset['return_scope'])
            if (tuple(core['fields']) != tuple(bound_reset['core']['fields'])
                    or tuple(tuple(f) for f in core['families']) != bound_reset['families']
                    or set(core['frame_ids']) & (set(before['frame_ids']) | set(fresh['frame_ids'])
                        | set(bound_reset['core']['frame_ids']))):
                raise LiveSelectionError('bound selection reset return state changed')
            event.update(accepted=True, reason='selection_cleared_without_business_submission',
                bound_selected_reset=True, returned_core={k: v for k, v in core.items() if k != 'candidates'})
            return events
        after, _ = bounded_checkpoint(None)
        event["after"] = {k:v for k,v in after.items() if k != 'candidates'}
        if after["selected_slot"] is not None or (not preview_only and any(
                after[k] != before[k] for k in ("families", "turns_remaining"))):
            raise LiveSelectionError("selection reset did not preserve hand and turn")
        event.update(accepted=True, reason="selection_cleared_without_business_submission")
        if inspected is not None:
            from live_decision.hif_inspected_art import (
                SELECTED_TITLE_SCOPE,
                INDEPENDENT_DETAIL_SCOPE,
                visible_art,
                independent_detail_reference_valid,
            )

            arts = [visible_art(frame, after['candidates'], inspect_slot) for frame in latest_frames]
            run = context.gkh_hif_live_context
            receipt = dict(run_id=run.run_id, revision=run.revision, slot=inspect_slot,
                family=inspected['family'], selected_frame_ids=inspected['selected_frame_ids'],
                reference_frame_ids=after['frame_ids'], detail=inspected['detail'], art=arts[0],
                title_confidence=min(row['score'] for item in inspected['detail']['observations']
                    for row in item['ocr'] if row.get('score', 0) >= .95))
            if inspected.get('title_scope') == SELECTED_TITLE_SCOPE:
                half_arts = fixed_left_half_arts(after)
                if half_arts:
                    arts, receipt['art'] = half_arts, half_arts[0]
                receipt['independent_detail'] = dict(scope=SELECTED_TITLE_SCOPE,
                    selected=inspected['selected'], before=event['before'], after=event['after'],
                    source_run_id=run.run_id, source_revision=run.revision,
                    cancel={k:event[k] for k in ('accepted', 'input_click_count', 'reason', 'target', 'capture_age_ms')},
                    returned_art=[dict(frame_id=frame_identifier(frame), art=art)
                        for frame, art in zip(latest_frames, arts, strict=True)])
                if not independent_detail_reference_valid(receipt, catalog=catalog):
                    raise LiveSelectionError('independent selected-title reference is incomplete', events=events)
            elif inspected.get('independent_ready') is True:
                # Choose the fixed identity region from its evidence contract,
                # not from whether the animated neighboring glow happens to
                # change during this particular returned capture batch.
                half_arts = fixed_left_half_arts(after)
                if half_arts:
                    candidate = dict(receipt, art=half_arts[0], independent_detail=dict(
                        scope=INDEPENDENT_DETAIL_SCOPE, selected=inspected['selected'],
                        before=event['before'], after=event['after'], source_run_id=run.run_id,
                        source_revision=run.revision,
                        cancel={k:event[k] for k in ('accepted', 'input_click_count', 'reason', 'target', 'capture_age_ms')},
                        returned_art=[dict(frame_id=frame_identifier(frame),art=art)
                            for frame,art in zip(latest_frames,half_arts,strict=True)]))
                    if independent_detail_reference_valid(candidate, catalog=catalog):
                        arts, receipt = half_arts, candidate
            if any(art != arts[0] for art in arts[1:]):
                raise LiveSelectionError('inspection reset returned artwork is not pixel stable', events=events)
            if 'independent_detail' not in receipt and any(
                    observation['predictions'][inspect_slot]['confidence'] < .85
                    or observation['predictions'][inspect_slot]['reason'] == 'below_acceptance_threshold'
                    for observation in after['identity_observations']):
                receipt['independent_detail'] = dict(scope=INDEPENDENT_DETAIL_SCOPE,
                    selected=inspected['selected'], before=event['before'], after=event['after'],
                    returned_art=[dict(frame_id=frame_identifier(frame),art=art)
                        for frame,art in zip(latest_frames,arts,strict=True)])
                if not independent_detail_reference_valid(receipt, catalog=catalog):
                    raise LiveSelectionError('independent inspection reference is incomplete', events=events)
            context.gkh_hif_inspected_art = receipt
            event['inspected_family'] = {k:v for k,v in receipt.items() if k != 'art'}
        return events
    except Exception as error:
        events.append({"phase": "selection_reset_stop", "accepted": False,
                       "reason": str(error), "input_click_count": 0})
        raise LiveSelectionError(str(error), events=events) from error


def _hif_slot_title_plate(candidate, plates):
    x, y, width, height = candidate.box
    matches = [p for p in plates if abs(p[0]-(x+2)) <= 1
        and p[1] >= y+height-36 and p[1]+p[3] <= y+height and p[2] <= width-2]
    return matches[0] if len(matches) == 1 else None


def _hif_base_title(text):
    from live_decision.upgrade import normalize_upgrade_title

    return normalize_upgrade_title(''.join(text.split())).rstrip('+')


def _hif_title_correction_view(prediction, hand_count):
    return (
        hand_count in (4, 5) and prediction.recognizer == 'card_art_embedding_top5_hif_masked_v1'
        and prediction.view_mode != 'full'
        or hand_count in (2, 3) and prediction.recognizer == 'card_art_embedding_top5_live_partial_v1'
        and prediction.view_mode == 'full' and prediction.visible_ratio == 1.0)


def _hif_weak_title_candidate(candidate, prediction, *, frame_id, minimum_margin, hand_count):
    """Only a current, calibrated weak artwork ranking earns title OCR."""
    scores, groups = prediction.top_k_scores, prediction.top_k_card_ids
    limit = min(.05, minimum_margin)
    return (
        _hif_title_correction_view(prediction, hand_count) and candidate.visible and prediction.visible
        and candidate.box[2:] == (190, 250) and prediction.box == candidate.box
        and prediction.slot == candidate.slot and prediction.frame_id == frame_id
        and prediction.source_frames == (frame_id,)
        and prediction.accepted is False
        and prediction.reason in ('below_acceptance_threshold', 'below_margin_threshold')
        and 2 <= len(scores) == len(groups) <= 5
        and all(groups) and all(math.isfinite(s) and -1 <= s <= 1 for s in scores)
        and all(a >= b for a, b in zip(scores, scores[1:]))
        and .70 <= scores[0] == prediction.confidence <= 1
        and 0 <= prediction.margin < limit
        and math.isclose(prediction.margin, scores[0]-scores[1], abs_tol=1e-6))


def _hif_title_visual_match(prediction, family, minimum_margin):
    groups, scores = prediction.top_k_card_ids, prediction.top_k_scores
    matches = [i for i, group in enumerate(groups) if family.intersection(group)]
    rank = matches[0] if len(matches) == 1 else None
    compatible = (rank is not None and set(groups[rank]) == family
                  and scores[rank] >= .70 and scores[0]-scores[rank] <= min(.05, minimum_margin))
    return rank, compatible


def _hif_current_title_family(context, frame, row, prediction, *, plan, minimum_margin, plates, catalog):
    """Read a current slot and identify the exact crop on a match or OCR miss."""
    from live_decision.inventory import _inventory_ocr

    slot = prediction.slot
    plate = _hif_slot_title_plate(row[slot], plates)
    if plate is None:
        return None, None
    x, y, width, height = plate
    right = min(x+width, row[slot+1].box[0]) if slot+1 < len(row) else x+width
    width = right-x
    if not (0 <= x < right <= frame.shape[1] and width >= 105):
        return None, None
    crop = frame[y:y+height, x:right].copy()
    roi = (x, y, width, height)
    title_pixels = (slot, roi, crop.tobytes())
    results = _inventory_ocr(context, crop, 'HifInventoryCompleteFamilyTitle', only_rec=False)
    if not results:
        return None, title_pixels
    if (len(results) != 1 or not _valid_ocr_result(results[0])
            or not isinstance(getattr(results[0], 'text', None), str)):
        return None, None
    result = results[0]
    if result.score < .95:
        return None, title_pixels
    tx, ty, tw, th = result.box
    # A detected word touching the crop edges can be a truncated name.
    if not (0 < tx < tx+tw < width and 0 < ty < ty+th < height):
        return None, None
    title = _hif_base_title(result.text)
    names = {key: _hif_base_title(entry.name) for key, entry in catalog.entries.items()
             if entry.plan in ('free', plan)}
    matching_ids = {key for key, name in names.items() if title == name}
    if (not title or not catalog.resolve_family(tuple(matching_ids), plan=plan)
            or (width < 180 and any(name != title and name.startswith(title) for name in names.values()))):
        return None, None
    groups, scores = prediction.top_k_card_ids, prediction.top_k_scores
    rank, compatible = _hif_title_visual_match(prediction, matching_ids, minimum_margin)
    return dict(scope='current_complete_title_family', accepted=compatible,
        reason='complete_base_title_and_visual_family_agree' if compatible else 'current_title_visual_family_conflict',
        frame_id=prediction.frame_id, slot=slot, roi=roi, text=result.text,
        confidence=result.score, family=tuple(groups[rank]) if compatible else tuple(sorted(matching_ids)),
        upgrade_verified=False,
        visual_candidate_rank=rank, visual_score_gap=scores[0]-scores[rank] if rank is not None else None,
        original_prediction=prediction.to_dict()), title_pixels


def _inventory_full_title_family(context, frame, candidate, prediction, *, minimum_margin,
                                 allow_low_margin_title=False, selected_neighbor_candidates=(),
                                 selected_slot=None, unselected_candidates=(), allow_inventory_title=False,
                                 allow_accepted_family_title=False):
    """Corroborate a proposed family with its entire visible base title.

    Navigation only needs the family. A covered plus suffix supplies no upgrade
    evidence; the complete inventory reader still establishes card variants.
    Ambiguous art requires explicit caller opt-in. Bound execution supplies
    an independently verified current hand before opting in, then consumes
    the title as separate family/margin evidence without rewriting the art.
    """
    from live_decision.hif_bound_title import binding, inventory_families

    known = binding(context, candidate.slot) if allow_accepted_family_title else None
    accepted_family_title = bool(known is not None and prediction.accepted
        and prediction.reason == 'accepted_visual_family'
        and prediction.top_k_card_ids and known['canonical_card_id'] in prediction.top_k_card_ids[0]
        and math.isfinite(prediction.margin) and prediction.margin >= max(.05, minimum_margin)
        and prediction.recognizer in ('card_art_embedding_top5_live_partial_v1',
            'card_art_embedding_top5_hif_masked_v1', 'card_art_embedding_full_masked_agreement_v1')
        and prediction.view_mode == 'full' and prediction.visible_ratio == 1.0
        and candidate.box[2:] == (190,250))
    inventory_title = (inventory_families(context)
        if allow_inventory_title and allow_low_margin_title and prediction.accepted is False
        and prediction.reason == 'below_acceptance_threshold'
        and prediction.recognizer == 'card_art_embedding_top5_live_partial_v1'
        and prediction.view_mode == 'full' and prediction.visible_ratio == 1.0
        and candidate.box[2:] == (190,250) and len(unselected_candidates) in (1,2,3) else None)
    bound_short_title = (binding(context, candidate.slot)
        if allow_low_margin_title and (accepted_family_title or prediction.accepted is False
        and prediction.reason == 'below_acceptance_threshold')
        and (prediction.recognizer == 'card_art_embedding_top5_live_partial_v1' or accepted_family_title)
        and prediction.view_mode == 'full' and prediction.visible_ratio == 1.0
        and candidate.box[2:] == (190,250) else None)
    complete_three_card_art = (allow_low_margin_title
        and (prediction.reason == 'below_margin_threshold' or bound_short_title is not None
             or inventory_title is not None)
        and (prediction.recognizer == 'card_art_embedding_top5_live_partial_v1' or accepted_family_title)
        and prediction.view_mode == 'full' and prediction.visible_ratio == 1.0
        and candidate.box[2:] == (190,250))
    navigation_masked_ambiguity = (allow_low_margin_title
        and prediction.recognizer == 'card_art_embedding_top5_hif_masked_v1'
        and prediction.reason in ('below_acceptance_threshold', 'below_margin_threshold')
        and math.isfinite(prediction.margin) and 0 <= prediction.margin < max(.05, minimum_margin))
    if (not isinstance(getattr(context, 'gkh_hif_live_context', None), HifLiveRunContext)
            or prediction.accepted and not accepted_family_title
            or not (prediction.recognizer == 'card_art_embedding_top5_hif_masked_v1'
                    or complete_three_card_art)
            or not prediction.top_k_card_ids or not math.isfinite(prediction.margin)
            or not ((prediction.reason == 'below_acceptance_threshold'
                     and prediction.margin >= max(.05, minimum_margin))
                    or (allow_low_margin_title and prediction.reason == 'below_margin_threshold'
                        and 0 <= prediction.margin < max(.05, minimum_margin))
                    or navigation_masked_ambiguity or bound_short_title is not None
                    or inventory_title is not None)
            or not math.isfinite(prediction.confidence) or not 0 <= prediction.confidence <= 1):
        return None
    from live_decision.inventory import _inventory_ocr
    from live_decision.hif_visual import _hif_unselected_title_plates, hif_unselected_row_candidates
    from live_decision.opening_observer import normalize_upgrade_title

    plates = _hif_unselected_title_plates(frame)
    selected_neighbor_geometry = None
    complete_short_card_row = False
    short_count = len(unselected_candidates)
    if (complete_three_card_art and (short_count in (1,2) or accepted_family_title and short_count in (4,5))
            and len(plates) == short_count):
        # A completed action leaves one or two full cards. Reuse the existing
        # complete unselected-row proof, not merely the number of title plates.
        proposed = hif_unselected_row_candidates(frame.shape, unselected_candidates, image=frame)
        complete_short_card_row = (proposed is not None and len(proposed) == short_count
            and tuple(c.box for c in proposed) == tuple(c.box for c in unselected_candidates)
            and tuple(c.slot for c in unselected_candidates) == tuple(range(short_count))
            and 0 <= candidate.slot < short_count and candidate == unselected_candidates[candidate.slot])
        if not complete_short_card_row:
            return None
    if complete_three_card_art and len(plates) != 3 and not complete_short_card_row:
        # 1214/1219: a selected card lifts its title out of the unselected strip.
        # The remaining titles need the entire two/three-card selected-row proof;
        # callers retain their independent exact SELECT gate.
        selected_hand_count = len(selected_neighbor_candidates)
        if (selected_hand_count not in (2,3) or len(plates) != selected_hand_count-1
                or type(selected_slot) is not int or not 0 <= selected_slot < selected_hand_count
                or candidate.slot == selected_slot):
            return None
        proposed = hif_selected_row_proposal(frame, selected_neighbor_candidates, hand_count=selected_hand_count)
        if (proposed is None or proposed[1] != selected_slot
                or tuple(c.box for c in proposed[0]) != tuple(c.box for c in selected_neighbor_candidates)):
            return None
        selected_neighbor_geometry = dict(selected_slot=selected_slot,
            candidate_boxes=[c.box for c in proposed[0]])
    plate = _hif_slot_title_plate(candidate, plates)
    if plate is None:
        return None
    px,py,pw,ph = plate
    rows = _inventory_ocr(context, frame[py:py+ph,px:px+pw].copy(),
        'HifInventoryCompleteFamilyTitle', only_rec=True)
    full_plate_title_ocr = None
    if (len(rows) == 1 and _valid_ocr_result(rows[0]) and 0 <= rows[0].score < .95
            and rows[0].box[0] >= 0 and rows[0].box[1] >= 0
            and rows[0].box[0]+rows[0].box[2] <= pw
            and rows[0].box[1]+rows[0].box[3] <= ph):
        # Native-size titles can have more background than text. Detect the
        # text inside this same complete plate once; require an independently
        # confident, fully contained reading of the identical entire title.
        detected = _inventory_ocr(context, frame[py:py+ph,px:px+pw].copy(),
            'HifInventoryCompleteFamilyTitle', only_rec=False)
        if (len(detected) == 1 and _valid_ocr_result(detected[0])
                and .95 <= detected[0].score <= 1
                and normalize_upgrade_title(''.join(detected[0].text.split()))
                    == normalize_upgrade_title(''.join(rows[0].text.split()))
                and detected[0].box[0] > 0 and detected[0].box[1] > 0
                and detected[0].box[2] > 0 and detected[0].box[3] > 0
                and detected[0].box[0]+detected[0].box[2] < pw
                and detected[0].box[1]+detected[0].box[3] < ph):
            full_plate_title_ocr = dict(text=rows[0].text, confidence=float(rows[0].score),
                box=list(rows[0].box), detected_box=list(detected[0].box))
            rows = detected
    if (len(rows) != 1 or not _valid_ocr_result(rows[0]) or not .95 <= rows[0].score <= 1
            or rows[0].box[0] < 0 or rows[0].box[1] < 0
            or rows[0].box[0]+rows[0].box[2] > pw or rows[0].box[1]+rows[0].box[3] > ph):
        return None
    text = _hif_base_title(rows[0].text)
    catalog = SkillCardFeatureCatalog.load(CATALOG_PATH)
    matching_ids = {key for key,entry in catalog.entries.items()
        if _hif_base_title(entry.name) == text}
    if accepted_family_title and matching_ids != set(prediction.top_k_card_ids[0]):
        return None
    family = set(prediction.top_k_card_ids[0])
    rank, score_gap = 0, 0.
    complete_full_art = None
    if complete_three_card_art:
        from live_decision.hif_bound_title import in_candidate_band

        if inventory_title is not None and (bound_short_title is None
                or bound_short_title['canonical_card_id'] not in matching_ids):
            # Fresh complete text and a frozen artwork candidate identify a
            # family after a draw. They do not inherit any pre-draw slot/order.
            groups = [set(ids) for ids in prediction.top_k_card_ids if set(ids) == matching_ids]
            if (not matching_ids or not matching_ids.intersection(inventory_title['canonical_card_ids'])
                    or len(groups) != 1 or not 1 <= len(prediction.top_k_card_ids) <= 5):
                return None
            rank = next(i for i, ids in enumerate(prediction.top_k_card_ids) if set(ids) == matching_ids)
            score_gap = (prediction.top_k_scores[0] - prediction.top_k_scores[rank]
                if len(prediction.top_k_scores) == len(prediction.top_k_card_ids) else None)
            family = matching_ids
            bound_short_title = None
        elif bound_short_title is not None:
            # The same complete current title and the already settled hand
            # identify this family even when score/support overlays corrupt
            # the embedding. This proves neither its enhancement nor input.
            if bound_short_title['canonical_card_id'] not in matching_ids:
                return None
            family = matching_ids
        elif not in_candidate_band(prediction, matching_ids, minimum_margin):
            return None
    if navigation_masked_ambiguity:
        # Navigation may disambiguate the frozen top-five candidates with a
        # complete, independently read base title. Keep the raw ranking intact;
        # this evidence never resolves enhancement or authorizes card input.
        scores = getattr(prediction, 'top_k_scores', ())
        if (not 1 <= len(scores) == len(prediction.top_k_card_ids) <= 5
                or any(not math.isfinite(score) for score in scores)
                or any(a < b for a,b in zip(scores, scores[1:]))):
            return None
        matches_by_rank = [(i, set(ids), scores[0] - scores[i])
            for i,ids in enumerate(prediction.top_k_card_ids)
            if matching_ids and matching_ids <= set(ids)
            and 0 <= scores[0] - scores[i] <= max(.05, minimum_margin)]
        if len(matches_by_rank) != 1:
            from live_decision.hif_bound_title import full_view_art_evidence

            complete_full_art = full_view_art_evidence(context, frame, unselected_candidates,
                prediction, matching_ids, minimum_margin=minimum_margin)
            if complete_full_art is None:
                return None
            rank, family, score_gap = None, matching_ids, None
        else:
            rank, family, score_gap = matches_by_rank[0]
    if not matching_ids or not matching_ids <= family:
        return None
    return dict(accepted=True, reason='complete_base_title_and_visual_family_agree',
        frame_id=frame_identifier(frame), slot=candidate.slot, roi=plate,
        text=rows[0].text, confidence=float(rows[0].score), family=sorted(family),
        visual_confidence=prediction.confidence, upgrade_verified=False,
        **({'full_plate_title_ocr': full_plate_title_ocr} if full_plate_title_ocr is not None else {}),
        **(dict(visual_candidate_rank=rank, visual_score_gap=score_gap,
                original_top_k_card_ids=prediction.top_k_card_ids,
                original_top_k_scores=prediction.top_k_scores,
                scope='navigation_family_only')
            if navigation_masked_ambiguity or complete_three_card_art else {}),
        **({'selected_neighbor_geometry': selected_neighbor_geometry}
           if selected_neighbor_geometry is not None else {}),
        **({'complete_unselected_hand_count': short_count} if complete_short_card_row else {}),
        **({'settled_hand_title_binding': bound_short_title} if bound_short_title is not None else {}),
        **({'accepted_visual_family_corroborated': True} if accepted_family_title else {}),
        **({'inventory_title_family': inventory_title} if inventory_title is not None
            and bound_short_title is None else {}),
        **({'complete_full_art': complete_full_art} if complete_full_art is not None else {}))


def _retain_hif_selected_neighbor_art(context, *, frames, rows, predictions, decision,
                                      generation, selected_slot, event, catalog=None):
    """Keep this accepted A's pixels for its one impending selection only."""
    if selected_slot is not None:
        return
    context.gkh_hif_selected_neighbor_art = None
    bundle = getattr(context, 'gkh_hif_opening_state', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    receipt = getattr(context, 'gkh_hif_inspected_art', None)
    if (selected_slot is not None or event.get('accepted') is not True
            or not decision.has_action_intent or bundle is None
            or not isinstance(run, HifLiveRunContext)
            or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)):
        return
    target = getattr(getattr(decision, 'action', None), 'card_slot', None)
    if type(target) is int and 0 <= target < len(bundle.state.deck.hand):
        from live_decision.hif_selected_neighbor_art import build_strong_hand_reference

        try:
            strong = build_strong_hand_reference(run_id=run.run_id,revision=run.revision,
                selected_slot=target,frames=frames,candidates=rows,predictions=predictions,
                generation=generation,hand_instance_ids=tuple(c.instance_id for c in bundle.state.deck.hand))
            context.gkh_hif_selected_neighbor_art = {'strong_hand': strong}
        except ValueError:
            pass
    if (type(target) is not int
            or not 0 <= target < target+1 < len(bundle.state.deck.hand)):
        return
    ref = bundle.state.deck.hand[target+1]
    definition = bundle.kernel.cards[ref.card_id]
    canonical = definition.canonical_card_id or ref.card_id
    inspected_source = (isinstance(receipt, dict) and receipt.get('slot') == target+1
        and canonical in receipt.get('family', ()))
    titles = [proof for proof in event.get('bound_complete_title_families', ())
        if proof.get('slot') == target+1 and canonical in proof.get('family', ())]
    title = titles[0] if not inspected_source and len(titles) == 1 else None
    if not inspected_source and title is None:
        return
    from live_decision.hif_selected_neighbor_art import build_reference

    try:
        reference = build_reference(run_id=run.run_id, revision=run.revision,
            selected_slot=target, card_instance_id=ref.instance_id, frames=frames,
            candidates=rows, prediction=predictions[target+1],
            inspection_reference=receipt if inspected_source else None, title_reference=title, catalog=catalog,
            generation=generation, hand_instance_ids=tuple(c.instance_id for c in bundle.state.deck.hand))
    except ValueError as error:
        event['selected_neighbor_reference_reason'] = str(error)
        return
    previous = context.gkh_hif_selected_neighbor_art or {}
    context.gkh_hif_selected_neighbor_art = dict(reference, **previous)
    event['selected_neighbor_reference'] = {key: reference[key] for key in (
        'scope', 'run_id', 'revision', 'selected_slot', 'slot', 'card_instance_id',
        'family', 'frame_ids', 'generation', 'hand_instance_ids', 'source_prediction')}
    if title is not None:
        event['selected_neighbor_reference']['title_source_frame_id'] = reference['title_source_frame_id']


def _hif_selected_neighbor_continuity(context, *, frames, rows, predictions, certificates,
                                      phase, selected_slot, expected_generation,
                                      selected_transition_verified):
    """Use exact current A/B artwork only inside the verified selection step."""
    bundle = getattr(context, 'gkh_hif_opening_state', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    reference = getattr(context, 'gkh_hif_selected_neighbor_art', None)
    if (phase != 'authoritative_refresh_b' or selected_transition_verified is not True
            or type(selected_slot) is not int or bundle is None or not isinstance(run, HifLiveRunContext)
            or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)
            or not 0 <= selected_slot < selected_slot+1 < len(bundle.state.deck.hand)
            or len(predictions) != len(bundle.state.deck.hand)):
        return None
    slot = selected_slot+1
    ref = bundle.state.deck.hand[slot]
    definition = bundle.kernel.cards[ref.card_id]
    canonical = definition.canonical_card_id or ref.card_id
    certificate = certificates.get(slot)
    current_plus = (certificate is not None and certificate.get('card_id') == definition.upgrade_card_id
        and any(fact.instance_id == ref.instance_id and fact.card_id == ref.card_id
                and fact.turn == bundle.state.turn and fact.projection_kind == 'ordinary_plus_estimate'
                for fact in bundle.state.hand_support))
    if (certificate is None or not (certificate.get('card_id') == canonical or current_plus)
            or not isinstance(reference, dict)
            or tuple(certificate.get('family', ())) != tuple(reference.get('family', ()))):
        return None
    from live_decision.hif_selected_neighbor_art import corroborate

    return corroborate(reference, frames=frames, candidates=rows, prediction=predictions[slot],
        run_id=run.run_id, revision=run.revision, selected_slot=selected_slot,
        card_instance_id=ref.instance_id, generation=expected_generation,
        hand_instance_ids=tuple(c.instance_id for c in bundle.state.deck.hand))


def _corroborate_retained_hif_families(context, frame, candidates, predictions, *,
                                    selected_slot, threshold, evidence):
    """Keep verified physical identities when the fresh row still agrees.

    This supports only the visual family, never a selected target, enhancement
    or input credential. Callers still verify current public fields and SELECT.
    """
    baseline = getattr(context, 'gkh_hif_committed_event', None)
    if not isinstance(baseline, dict) or len(predictions) != len(candidates):
        return predictions
    try:
        ids = _retained_hif_hand_ids(context, baseline['core']['families'])
    except (LiveSelectionError, ValueError, KeyError, AttributeError):
        return predictions
    fid = frame_identifier(frame)
    if (len(ids) != len(predictions) or fid in baseline['core']['frame_ids']
            or any(p.slot != i or c.slot != i or p.frame_id != fid
                or tuple(p.box) != tuple(c.box) or not p.top_k_card_ids
                or ids[i] not in p.top_k_card_ids[0]
                for i, (c, p) in enumerate(zip(candidates, predictions, strict=True)))):
        return predictions
    result = list(predictions)
    for slot, prediction in enumerate(predictions):
        if (slot != selected_slot and prediction.accepted is False
                and prediction.reason == 'below_margin_threshold'
                and math.isfinite(prediction.confidence) and threshold <= prediction.confidence <= 1
                and math.isfinite(prediction.margin) and 0 <= prediction.margin <= 1):
            evidence.append(dict(scope='retained_same_hand_family', slot=slot,
                card_id=ids[slot], frame_id=fid, source_frame_ids=baseline['core']['frame_ids'],
                original_prediction=prediction.to_dict(), upgrade_verified=False))
            result[slot] = replace(prediction, accepted=True, reason='retained_same_hand_family')
    return tuple(result)


def _retained_hif_family_margin(context, certificate, prediction, event, minimum_margin):
    """Consume the resolved same-hand family without upgrading its raw score."""
    if not prediction.accepted or prediction.slot == event.get('expected_selected_slot'):
        return None
    baseline = getattr(context, 'gkh_hif_committed_event', None)
    try:
        ids = _retained_hif_hand_ids(context, baseline['core']['families'])
        proofs = [p for p in event.get('retained_hand_families', ()) if p.get('slot') == prediction.slot]
        if len(proofs) > 1:
            return None
        if proofs:
            proof = proofs[0]
        elif prediction.margin >= minimum_margin:
            # A fresh independently accepted family also confirms this hand.
            # Its stronger margin must not be vetoed by the old weak probe.
            proof = dict(scope='retained_same_hand_family',slot=prediction.slot,
                card_id=ids[prediction.slot],frame_id=prediction.frame_id,
                source_frame_ids=baseline['core']['frame_ids'],original_prediction=prediction.to_dict(),
                upgrade_verified=False)
        else:
            return None
        raw = proof['original_prediction']
        generation = dict(event['generation'])
        if event.get('selected_score_preview_delta'):
            generation['score'] = event['expected_generation']['score']
        fields = [generation[key] for key in
            ('turns_remaining','stamina','shield','score','score_multiplier','action_order')]
        current = prediction.to_dict()
        if (proof.get('scope') != 'retained_same_hand_family'
                or proof.get('card_id') != ids[prediction.slot]
                or certificate['card_id'] not in prediction.top_k_card_ids[0]
                or tuple(certificate['family']) != tuple(prediction.top_k_card_ids[0])
                or proof.get('frame_id') != prediction.frame_id or prediction.frame_id != event['frame_ids'][-1]
                or tuple(proof.get('source_frame_ids', ())) != tuple(baseline['core']['frame_ids'])
                or json.dumps(fields) != json.dumps(baseline['core']['fields'])
                or generation['turn'] != context.gkh_hif_opening_state.state.turn
                or generation['hand_count'] != len(ids)
                or any(current[key] != value for key,value in raw.items() if key not in ('accepted','reason'))
                or proofs and (raw.get('accepted') is not False or raw.get('reason') != 'below_margin_threshold')):
            return None
        return proof  # Family only; target details, resources and input time remain independent.
    except (LiveSelectionError, ValueError, KeyError, TypeError, AttributeError, IndexError):
        return None


def _corroborate_bound_hif_families(context, frame, candidates, predictions, *,
                                  selected_slot, minimum_margin, evidence, selected_neighbor_frames=(),
                                  minimum_confidence=None, selected_neighbor_candidates=None):
    """Use a full current title plus the bound permanent identity, never a guess.

    The same-frame title only proves the visual family. Ordinary enhancement,
    selected effects, resources and final input freshness keep their own gates.
    """
    bundle = getattr(context, 'gkh_hif_opening_state', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    if (bundle is None or not isinstance(run, HifLiveRunContext)
            or (bundle.run_id,bundle.revision) != (run.run_id,run.revision)
            or len(candidates) != len(predictions)
            or len(candidates) != len(bundle.state.deck.hand)):
        return predictions
    from live_decision.hif_bound_title import SCOPE, binding
    from live_decision.hif_inspected_art import corroborate

    retained_evidence = []
    predictions = _corroborate_retained_hif_families(context, frame, candidates, predictions,
        selected_slot=selected_slot, threshold=minimum_confidence if minimum_confidence is not None else .85,
        evidence=retained_evidence)
    if retained_evidence:
        evidence.setdefault('retained_hand_families', []).extend(retained_evidence)
    inspected_evidence = []
    inspected = getattr(context, 'gkh_hif_inspected_art', None)
    inspected_slot = inspected.get('slot', -1) if inspected else -1
    if 0 <= inspected_slot < len(bundle.state.deck.hand):
        inspected_ref = bundle.state.deck.hand[inspected_slot]
        inspected_definition = bundle.kernel.cards[inspected_ref.card_id]
        inspected_id = inspected_definition.canonical_card_id or inspected_ref.card_id
        inspected_keys = [key for key in bundle.inventory_keys if key[0] == inspected_ref.instance_id]
        inspected_bound = (inspected_id in inspected['family'] and len(inspected_keys) == 1
                           and inspected_keys[0][1] == inspected_id)
    else:
        inspected_bound = False
    if inspected_bound:
        result = list(corroborate(context, frame, candidates, predictions,
            selected_slot=selected_slot, evidence=inspected_evidence))
    else:
        from live_decision.hif_opening_art import corroborate as opening_corroboration

        result = list(opening_corroboration(context, frame, candidates, predictions,
            selected_slot=selected_slot, evidence=inspected_evidence))
    reference = (getattr(context, 'gkh_hif_selected_neighbor_art', None) or {}).get('strong_hand')
    complete = getattr(context, 'gkh_hif_committed_event', None)
    if (reference is not None and selected_slot is not None and len(selected_neighbor_frames) == 3
            and isinstance(complete, dict) and complete.get('accepted') is True
            and complete.get('checks') and all(v is True for v in complete['checks'].values())
            and any(not p.accepted and p.slot != selected_slot for p in predictions)):
        from live_decision.hif_selected_neighbor_art import corroborate_strong_hand

        frames = tuple(selected_neighbor_frames)
        # The caller has just detected and reconciled these exact three frames.
        # Reuse only that call's rows; the artwork proof still checks every
        # frame, instance, physical box and matching-frame input deadline.
        rows = (tuple(selected_neighbor_candidates) if selected_neighbor_candidates is not None else
                tuple(_detect_live_candidates(context, image) for image in frames))
        prepared = context.gkh_hif_card_art
        batches = tuple(predictions if frame_identifier(image) == frame_identifier(frame) else
            classify_live_card_views(prepared.recognizer,image,row,frame_id=frame_identifier(image),
                plan='sense',acceptance_threshold=max(.85,minimum_confidence or .85),min_margin=minimum_margin,
                **_live_card_art_profile(context,row,selected_slot))
            for image,row in zip(frames,rows,strict=True))
        state = bundle.state
        generation = dict(turn=state.turn,turns_remaining=state.turns_remaining,stamina=state.stamina,
            shield=state.genki,score=state.score,score_multiplier=state.score_multiplier,
            hand_count=len(state.deck.hand),plan='sense',action_order=list(complete['core']['fields'][5]))
        proofs = corroborate_strong_hand(reference,run_id=run.run_id,revision=run.revision,
            hand_instance_ids=tuple(c.instance_id for c in state.deck.hand),generation=generation,
            selected_slot=selected_slot,frames=frames,candidates=rows,predictions=batches)
        for proof in proofs:
            slot = proof['slot']
            if result[slot].accepted:
                continue
            evidence.setdefault('selection_common_art_families',[]).append(proof)
            matched = proof.get('matched_prediction_frame_id')
            if matched is not None:
                current = next(batch[slot] for batch in batches if batch[slot].frame_id == matched)
                result[slot] = replace(current,reason='bound_selection_common_art')
            else:
                source = proof['source_prediction']
                result[slot] = replace(result[slot],accepted=True,card_id=source['card_id'],
                    confidence=source['confidence'],margin=source['margin'],reason='bound_selection_common_art')
    if inspected_evidence:
        for proof in inspected_evidence:
            key = ('opening_exact_art_families' if proof.get('scope') == 'verified_opening_exact_current_art_v1'
                else 'inspected_title_art_families')
            evidence.setdefault(key, []).append(proof)
    for index,(candidate,prediction) in enumerate(zip(candidates,predictions,strict=True)):
        slot = candidate.slot
        if slot != index or prediction.slot != slot or slot == selected_slot:
            continue
        if result[index].reason in ('bound_selection_common_art', 'verified_opening_exact_current_art',
                'retained_same_hand_family', 'inspected_title_same_hand'):
            continue
        family = tuple(prediction.top_k_card_ids[0]) if prediction.top_k_card_ids else ()
        ref = bundle.state.deck.hand[slot]
        definition = bundle.kernel.cards[ref.card_id]
        canonical = definition.canonical_card_id or ref.card_id
        keys = [key for key in bundle.inventory_keys if key[0] == ref.instance_id]
        if len(keys) != 1 or keys[0][1] != canonical:
            continue
        bound = binding(context, slot)
        covered_neighbor = selected_slot is not None and slot == selected_slot+1
        if covered_neighbor and bound is None:
            continue
        if canonical not in family and bound is None:
            continue
        complete_accepted_title = (bound is not None and minimum_confidence is not None
            and prediction.accepted and prediction.confidence < minimum_confidence)
        proof = _inventory_full_title_family(context,frame,candidate,prediction,
            allow_accepted_family_title=complete_accepted_title,
            **(dict(unselected_candidates=candidates)
                if bound is not None and len(candidates) in (1,2,3,4,5) and selected_slot is None else {}),
            **(dict(selected_neighbor_candidates=candidates,selected_slot=selected_slot)
                if bound is not None and len(candidates) in (2,3) and selected_slot is not None else {}),
            minimum_margin=minimum_margin, **({'allow_low_margin_title': True}
                if bound is not None and (complete_accepted_title or prediction.recognizer ==
                'card_art_embedding_top5_hif_masked_v1' or len(candidates) in (1,2,3)
                and prediction.recognizer == 'card_art_embedding_top5_live_partial_v1') else {}))
        if proof is None or canonical not in proof['family']:
            continue
        if covered_neighbor and proof.get('scope') != 'navigation_family_only':
            # A high-margin but weak masked neighbor can have an entirely
            # readable title. Establish it on this observation's three frames;
            # the navigation-only single-frame proof is not an input credential.
            frames = tuple(selected_neighbor_frames)
            frame_ids = tuple(frame_identifier(image) for image in frames)
            if (len(frames) != 3 or len(set(frame_ids)) != 3
                    or prediction.frame_id != frame_identifier(frame)
                    or prediction.frame_id not in frame_ids
                    or prediction.recognizer != 'card_art_embedding_top5_hif_masked_v1'
                    or prediction.reason != 'below_acceptance_threshold'
                    or prediction.view_mode != 'left_occluded'
                    or prediction.margin < max(.05, minimum_margin)):
                continue
            observations = []
            for image in frames:
                proposed = hif_selected_row_proposal(image, candidates, hand_count=len(candidates))
                if (proposed is None or proposed[1] != selected_slot
                        or tuple(c.box for c in proposed[0]) != tuple(c.box for c in candidates)):
                    break
                title = proof if frame_identifier(image) == prediction.frame_id else _inventory_full_title_family(
                    context, image, candidate, prediction, minimum_margin=minimum_margin)
                if title is None or title['family'] != proof['family'] or title['text'] != proof['text']:
                    break
                observations.append(title)
            if len(observations) != 3:
                continue
            proof = dict(proof, **bound, scope=SCOPE,
                confidence=min(row['confidence'] for row in observations),
                selected_neighbor_title_observations=observations)
        if proof.get('scope') == 'navigation_family_only':
            if bound is None or prediction.frame_id != frame_identifier(frame):
                continue
            proof = dict(proof, **bound, scope=SCOPE)
        proof = dict(proof,
            run_id=bundle.run_id,revision=bundle.revision,card_instance_id=ref.instance_id,
            original_prediction=prediction.to_dict())
        result[index] = replace(prediction,accepted=True,
            confidence=proof['confidence'],reason='bound_complete_title_and_visual_family')
        if bound is not None and proof.get('scope') is None:
            from live_decision.hif_bound_title import bind_complete_base_title

            proof = bind_complete_base_title(context,result[index],proof) or proof
        evidence.setdefault('bound_complete_title_families',[]).append(proof)
    return tuple(result)


def _corroborate_bound_hif_selected_family(context, frames, predictions, certificates, *,
                                          selected_slot, minimum_margin, evidence, catalog, plan,
                                          identity_frame_id=None, selected_detail=None, candidates=(),
                                          minimum_confidence=None):
    """A previously corroborated card still needs a fresh complete selected title.

This does not carry the unselected pixels across selection. The selected frame's
own candidate band and all three complete detail titles must independently agree.
"""
    from live_decision.hif_bound_title import SCOPE, binding, valid_proof, in_candidate_band

    if selected_slot is None or len(frames) != 3:
        return predictions
    frame_ids = tuple(frame_identifier(frame) for frame in frames)
    if identity_frame_id is None:
        identity_frame_id = frame_ids[-1]
    elif len(set(frame_ids)) != 3 or identity_frame_id not in frame_ids:
        return predictions
    matches = [row for row in certificates if row['slot'] == selected_slot]
    if len(matches) != 1 or not 0 <= selected_slot < len(predictions):
        return predictions
    certificate = matches[0]
    source = certificate.get('source_prediction')
    source_proof = certificate.get('bound_title_evidence')
    if isinstance(source, dict) and isinstance(source_proof, dict):
        from types import SimpleNamespace

        raw_source = source
        source = SimpleNamespace(**raw_source, to_dict=lambda: raw_source)
        source_proof = dict(source_proof)
        source_proof['hand_instance_ids'] = tuple(source_proof.get('hand_instance_ids', ()))
        key = source_proof.get('inventory_key', ())
        if len(key) != 3:
            return predictions
        source_proof['inventory_key'] = (key[0], key[1], tuple(tuple(x) for x in key[2]))
    if source is None or not callable(getattr(source, 'to_dict', None)):
        return predictions
    prediction = predictions[selected_slot]
    bound = binding(context, selected_slot)
    family = tuple(certificate['family'])
    from live_decision.hif_inspected_art import source_title_evidence_valid

    inspected_source = selected_detail is not None and source_title_evidence_valid(context, certificate)
    accepted_family_title = bool(minimum_confidence is not None and prediction.accepted
        and prediction.reason == 'accepted_visual_family' and prediction.confidence < minimum_confidence
        and prediction.top_k_card_ids and tuple(prediction.top_k_card_ids[0]) == family
        and prediction.margin >= max(.05, minimum_margin))
    strong_source = bool(accepted_family_title and source.accepted
        and source.confidence >= minimum_confidence and source.margin >= max(.05, minimum_margin)
        and source.raw_score >= minimum_confidence
        and source.top_k_card_ids and tuple(source.top_k_card_ids[0]) == family)
    if (bound is None or not (valid_proof(context, source, source_proof) or inspected_source or strong_source)
            or prediction.frame_id != identity_frame_id):
        return predictions
    detail = selected_detail
    visible_id = bound['canonical_card_id']
    bundle = context.gkh_hif_opening_state
    ref = bundle.state.deck.hand[selected_slot]
    if any(fact.instance_id == ref.instance_id and fact.card_id == ref.card_id
           and fact.turn == bundle.state.turn and fact.projection_kind == 'ordinary_plus_estimate'
           for fact in getattr(bundle.state, 'hand_support', ())):
        visible_id = bundle.kernel.cards[ref.card_id].upgrade_card_id
    complete_full_art = singleton_titles = None
    if not accepted_family_title and not in_candidate_band(prediction, family, minimum_margin):
        from live_decision.hif_bound_title import full_view_art_evidence, selected_singleton_title_evidence

        image = frames[frame_ids.index(identity_frame_id)]
        complete_full_art = full_view_art_evidence(context, image, candidates, prediction,
            family, minimum_margin=minimum_margin, selected_slot=selected_slot)
        if complete_full_art is None:
            if len(candidates) != 1 or prediction.accepted:
                return predictions
            if detail is None:
                detail = _hif_selected_detail_identity(context, frames, family, catalog, plan)
            singleton_titles = selected_singleton_title_evidence(context, frames, candidates,
                prediction, detail, family_ids=family, card_id=visible_id, catalog=catalog, plan=plan)
            if singleton_titles is None:
                return predictions
    if detail is None:
        detail = _hif_selected_detail_identity(context, frames, family, catalog, plan)
    if (detail['accepted'] is not True or detail['card_id'] != visible_id
            or visible_id not in family
            or certificate['card_id'] not in (bound['canonical_card_id'], visible_id)):
        return predictions
    if inspected_source or accepted_family_title:
        from live_decision.hif_bound_title import selected_detail_family_valid

        if not selected_detail_family_valid(detail, frame_ids=frame_ids, family_ids=family,
                card_id=visible_id if accepted_family_title else certificate['card_id'], catalog=catalog, plan=plan):
            return predictions
    confidence = min(row['score'] for observation in detail['observations']
        for row in observation['ocr'] if row.get('score', 0) >= .95)
    proof = dict(bound, scope=SCOPE, family=family, frame_id=prediction.frame_id,
        accepted=True, confidence=confidence, upgrade_verified=False,
        reason='bound_fresh_selected_complete_title', selected_detail=detail,
        original_prediction=prediction.to_dict(),
        **({'complete_full_art': complete_full_art} if complete_full_art is not None else {}),
        **({'complete_singleton_titles': singleton_titles} if singleton_titles is not None else {}),
        **({'accepted_visual_family_corroborated': True} if accepted_family_title else {}),
        **({'source_identity_kind': 'inspected_art'} if inspected_source else {}))
    evidence.setdefault('bound_complete_title_families', []).append(proof)
    result = list(predictions)
    result[selected_slot] = replace(prediction, accepted=True,
        confidence=prediction.confidence if inspected_source else confidence,
        reason='bound_fresh_selected_complete_title')
    return tuple(result)


def _corroborate_recovered_support_family(context, frames, predictions, *, baseline,
                                         selected_slot, minimum_margin, evidence):
    """Use this same observation's reverified receipt, never the saved selection."""
    from live_decision.hif_bound_title import SCOPE, binding, in_candidate_band
    from live_decision.hif_gray_selection import recovered_selected_support_hand_identity

    if (selected_slot is None or not isinstance(baseline, dict)
            or baseline.get('selected_skill_support') is None
            or not 0 <= selected_slot < len(predictions)):
        return predictions
    prediction = predictions[selected_slot]
    bound = binding(context, selected_slot)
    if prediction.accepted or bound is None:
        return predictions
    families = [ids for ids in prediction.top_k_card_ids if bound['canonical_card_id'] in ids]
    frame_ids = tuple(frame_identifier(frame) for frame in frames)
    if (len(families) != 1 or prediction.frame_id not in frame_ids
            or not in_candidate_band(prediction, families[0], minimum_margin)):
        return predictions
    identity = recovered_selected_support_hand_identity(context.gkh_hif_opening_state,
        complete_event=context.gkh_hif_committed_event,
        journal=context.gkh_hif_gray_selected_source, baseline=baseline,
        frame_ids=frame_ids, slot=selected_slot)
    if (identity['card_id'] != bound['canonical_card_id']
            or identity['card_instance_id'] != bound['card_instance_id']):
        raise ValueError('fresh selected support family differs from its bound entity')
    receipt = baseline['selected_skill_support']
    confidence = min(row['title_rows'][0]['confidence'] for row in receipt['observations'])
    proof = dict(bound, scope=SCOPE, family=tuple(families[0]), frame_id=prediction.frame_id,
        accepted=True, confidence=confidence, upgrade_verified=False,
        reason='bound_fresh_selected_support_receipt', selected_support_identity=identity,
        selected_support=receipt, original_prediction=prediction.to_dict())
    evidence.setdefault('bound_complete_title_families', []).append(proof)
    result = list(predictions)
    result[selected_slot] = replace(prediction, accepted=True, confidence=confidence,
        reason='bound_fresh_selected_support_receipt')
    return tuple(result)


def _inventory_live_checkpoint(context, frames, recognizer, *, identity_threshold,
                               identity_min_margin, single_frame=True, return_observation=None,
                               settlement_survivors=None, inventory_before=None):
    """Observe the unchanged hand and visible resources around a page visit."""
    run=getattr(context,'gkh_hif_live_context',None)
    expected_empty=(isinstance(run,HifLiveRunContext)
        and getattr(context,'gkh_hif_expected_empty_hand',None)==(run.run_id,run.revision)
        and getattr(context,'gkh_hif_opening_state',None) is not None)
    if len(frames) != 1:
        raise ValueError('inventory checkpoint requires the current frame')
    if expected_empty:
        from live_decision.inventory import _inventory_ocr
        candidates=[]
        for frame in frames:
            detail=context.run_recognition('ProduceRecognitionCards',frame)
            row=_normalise_observed_live_candidates(context,frame,
                list(detail.all_results) if detail and detail.hit else [])
            # The neural model may label a bottom drink/control decoration as
            # a card. Only the calibrated hand strip participates in this
            # zero-hand check, still corroborated by the explicit UI sentence.
            row=tuple(c for c in row if 704<=c.box[1] and c.box[1]+c.box[3]<=1160)
            labels=_inventory_ocr(context,frame,'HifEmptyHandLabel',roi=[188,984,346,42],only_rec=True)
            if (row or len(labels)!=1 or not .95<=labels[0].score<=1
                    or re.sub(r'\s+','',labels[0].text)!='手札のスキルカードが0枚です'):
                failure_sink=getattr(context,'gkh_hif_inventory_failure_sink',None)
                if callable(failure_sink):
                    failure_sink(frame)
                same_label=(not row and len(labels)==1
                    and re.sub(r'\s+','',labels[0].text)=='手札のスキルカードが0枚です')
                raise LiveSelectionError('empty hand needs explicit zero-card label and no card candidates',events=[dict(
                    phase='inventory_live_geometry' if same_label else 'inventory_empty_hand_contradiction',
                    geometry_stable=False,frame_id=frame_identifier(frame),
                    labels=[dict(text=r.text,confidence=r.score) for r in labels],
                    candidates=[dict(box=c.box,label=c.detector_label,confidence=c.detector_score) for c in row])])
            candidates.append(())
        candidates,evidence=tuple(candidates),[]
    else:
        detail = context.run_recognition("ProduceRecognitionCards", frames[0])
        row = _normalise_observed_live_candidates(context, frames[0],
            list(detail.all_results) if detail and detail.hit else [])
        if not row or _locate_selected_live_slots(context, frames[0], row):
            raise LiveSelectionError("inventory fresh recheck is not an unselected hand")
        candidates, evidence = (row,), []
    families, fields, title_corroboration = [], [], []
    art_preprocessing_evidence = []
    return_predictions, unresolved_target = [], False
    frame_ids = tuple(frame_identifier(frame) for frame in frames)
    current_return = return_observation is not None
    if current_return and expected_empty:
        raise LiveSelectionError('inventory inspected return lacks the current hand observation')
    current_title_scope = (isinstance(run, HifLiveRunContext) and not expected_empty
        and inventory_before is None and settlement_survivors is None)
    current_title_catalog = None
    current_family_rows, corrected_slots = [], set()
    title_reads, pending_titles = {}, []
    names = ("turns_remaining", "stamina", "shield", "score", "score_multiplier", "action_order")
    for frame, frame_id, row in zip(frames, frame_ids, candidates, strict=True):
        title_families = {}
        art_row = row
        if getattr(context, 'gkh_hif_upstream_card_ui', False) is True:
            from live_decision.hif_upstream_policy import unselected_art_candidates

            try:
                art_row = unselected_art_candidates(row, tuple(frame.shape), image=frame)
            except ValueError as error:
                # Detector halos are transient geometry, not a fatal business
                # mismatch. Request another complete group under the existing
                # eight-group budget; retain no fields from this rejected set.
                raise LiveSelectionError(str(error), events=[dict(
                    phase='inventory_live_geometry', geometry_stable=False,
                    frame_ids=[frame_identifier(f) for f in frames],
                    candidates=[list(c.box) for c in row])]) from error
            if art_row != row:
                art_preprocessing_evidence.append(dict(source='upstream_unselected_art_footprint',
                    frame_id=frame_identifier(frame), original_boxes=[list(c.box) for c in row],
                    artwork_boxes=[list(c.box) for c in art_row]))
        predictions = () if expected_empty else classify_live_card_views(recognizer, frame, art_row,
            frame_id=frame_id, plan=_infer_plan(context),
            acceptance_threshold=identity_threshold, min_margin=identity_min_margin,
            art_preprocessing_evidence=art_preprocessing_evidence,
            **_live_card_art_profile(context, art_row, None))
        raw_predictions = predictions
        current_titles, title_misses, current_families = {}, {}, {}
        current_family_rows.append(current_families)
        if (current_title_scope and frame.shape == (1280, 720, 3)
                and tuple(c.slot for c in art_row) == tuple(range(len(art_row)))
                and all(a.box[0] < b.box[0] for a, b in zip(art_row, art_row[1:]))):
            weak = [p for c, p in zip(art_row, raw_predictions, strict=True)
                    if _hif_weak_title_candidate(c, p, frame_id=frame_id,
                        minimum_margin=identity_min_margin, hand_count=len(art_row))]
            if weak:
                from live_decision.hif_visual import _hif_unselected_title_plates

                plates = _hif_unselected_title_plates(frame)
                if current_title_catalog is None:
                    current_title_catalog = SkillCardFeatureCatalog.load(CATALOG_PATH)
                for prediction in weak:
                    proof, title_pixels = _hif_current_title_family(context, frame, art_row, prediction,
                        plan=run.plan, minimum_margin=identity_min_margin, plates=plates, catalog=current_title_catalog)
                    if proof is None:
                        if title_pixels is not None:
                            title_misses[prediction.slot] = (title_pixels, prediction)
                        continue
                    if not proof['accepted']:
                        raise LiveSelectionError('current Live card title conflicts with visual candidates', events=[dict(
                            phase='inventory_live_identity', frame_ids=frame_ids, title_family=proof,
                            click_dispatched=False)])
                    current_titles[prediction.slot] = proof['family']
                    title_reads.setdefault(title_pixels, proof)
                    title_corroboration.append(proof)
                    if proof['visual_candidate_rank'] > 0:
                        corrected_slots.add(prediction.slot)
            current_families.update({p.slot: current_titles.get(p.slot, (
                tuple(p.top_k_card_ids[0]) if p.accepted and p.top_k_card_ids and p.frame_id == frame_id
                and p.source_frames == (frame_id,)
                and p.slot == c.slot and p.box == c.box else None))
                for c, p in zip(art_row, raw_predictions, strict=True)})
        from live_decision.hif_hand_inspection import corroborate_upstream_inspection

        predictions = corroborate_upstream_inspection(context, predictions, title_corroboration)
        # Only opening, browsing and closing this inventory have occurred.
        # A weak first choice can be corroborated by the same physical slot
        # in the accepted hand immediately before the visit.
        if (inventory_before is not None and len(predictions) == len(inventory_before['families'])
                and all(p.slot == i and p.top_k_card_ids
                        and tuple(p.top_k_card_ids[0]) == inventory_before['families'][i]
                        for i, p in enumerate(predictions))):
            current = list(predictions)
            for i, prediction in enumerate(predictions):
                if not prediction.accepted and prediction.reason in (
                        'below_acceptance_threshold', 'below_margin_threshold'):
                    title_corroboration.append(dict(scope='same_slot_after_inventory', slot=i,
                        source_frame_ids=inventory_before['frame_ids'], frame_id=prediction.frame_id,
                        family=inventory_before['families'][i], original_prediction=prediction.to_dict(),
                        upgrade_verified=False))
                    current[i] = replace(prediction, accepted=True, reason='same_slot_after_inventory')
            predictions = tuple(current)
        if return_observation is not None:
            expected = return_observation['families']
            if (len(predictions) != len(expected) or any(
                    p.accepted and (not p.top_k_card_ids or tuple(p.top_k_card_ids[0]) != expected[i])
                    for i, p in enumerate(predictions))):
                raise LiveSelectionError('inventory return visual family changed', events=[dict(
                    phase='inventory_inspected_return_contradiction', click_dispatched=False)])
            return_predictions.append(raw_predictions)
        from live_decision.hif_inspected_art import corroborate

        predictions = corroborate(context, frame, row, predictions,
            selected_slot=None, evidence=title_corroboration)
        from live_decision.hif_inspected_art import inspected_family

        for prediction in predictions:
            for proof in title_corroboration:
                known = inspected_family(prediction, proof)
                if known:
                    title_families[prediction.slot] = known
        if any(not prediction.accepted for prediction in predictions):
            resolved = []
            # The neural detector can crop the right edge of an overlapping
            # card. Locate its visible title inside the same physical footprint
            # already used for its artwork, keeping the raw input boxes intact.
            for candidate, prediction in zip(art_row, predictions, strict=True):
                proof = None if candidate.slot in current_titles else _inventory_full_title_family(
                    context, frame, candidate, prediction,
                    minimum_margin=identity_min_margin, allow_low_margin_title=True,
                    unselected_candidates=art_row, allow_inventory_title=True)
                if (proof is not None and current_title_scope
                        and _hif_title_correction_view(prediction, len(art_row))
                        and proof.get('visual_candidate_rank', 0) not in (0, None)
                        and not proof.get('inventory_title_family')):
                    # A legacy title read alone cannot change the current artwork rank.
                    proof = None
                if proof is not None:
                    title_corroboration.append(proof)
                    title_families[candidate.slot] = tuple(proof['family'])
                    prediction = replace(prediction, accepted=True,
                        reason='inventory_complete_title_family_corroborated')
                resolved.append(prediction)
            predictions = tuple(resolved)
        title_families.update(current_titles)
        unresolved = [p.slot for p in predictions
                      if p.slot not in current_titles and (not p.accepted or not p.top_k_card_ids)]
        diagnostic_target = (return_observation is not None
            and unresolved == [return_observation['slot']]
            and raw_predictions[return_observation['slot']].accepted is False
            and raw_predictions[return_observation['slot']].reason in (
                'below_acceptance_threshold', 'below_margin_threshold'))
        if unresolved and not diagnostic_target:
            defer_title_misses = all(slot in title_misses for slot in unresolved)
            failure_sink = getattr(context, 'gkh_hif_inventory_failure_sink', None)
            if not defer_title_misses and callable(failure_sink):
                for failed_frame in frames:
                    failure_sink(failed_frame)
            failure = dict(
                phase='inventory_live_identity', frame_id=frame_identifier(frame),
                art_preprocessing_evidence=art_preprocessing_evidence,
                frame_ids=[frame_identifier(f) for f in frames],
                predictions=[dict(slot=p.slot,accepted=p.accepted,reason=p.reason,confidence=p.confidence,
                    margin=p.margin,recognizer=p.recognizer,
                    top_k_card_ids=p.top_k_card_ids,box=p.box) for p in predictions])
            if defer_title_misses:
                pending_titles.append((len(families), failure, [title_misses[slot] for slot in unresolved]))
            else:
                raise LiveSelectionError("inventory navigation Live visual family is unresolved", events=[failure])
        unresolved_target |= diagnostic_target
        # Keep missing identity evidence unresolved while reading the checkpoint.
        families.append(tuple(None if p.slot in unresolved else
            title_families.get(p.slot, tuple(p.top_k_card_ids[0])) for p in predictions))
        if current_return and any(family is not None and family != expected for family, expected in
                zip(families[-1], return_observation['families'], strict=True)):
            raise LiveSelectionError('inventory return visual family changed', events=[dict(
                phase='inventory_inspected_return_contradiction', frame_id=frame_id, click_dispatched=False)])
        observed = _observe_live_fields(context, frame, commit_turn=False)
        shield_confidence = observed['shield'].get('confidence')
        if (isinstance(run, HifLiveRunContext) and observed['shield'].get('accepted') is True
                and (type(shield_confidence) not in (int, float)
                     or not math.isfinite(shield_confidence) or not .95 <= shield_confidence <= 1)):
            # Parser acceptance proves one numeric value, not OCR certainty.
            # A low-confidence prefix must not become stable checkpoint truth.
            observed['shield'] = dict(observed['shield'], accepted=False,
                reason='hif_checkpoint_shield_confidence_below_gate')
        if any(observed[name].get("accepted") is not True for name in names):
            failure_sink=getattr(context,'gkh_hif_inventory_failure_sink',None)
            if callable(failure_sink):
                failure_sink(frame)
            raise LiveSelectionError("inventory navigation Live resources are unresolved", events=[{
                "phase": "inventory_live_checkpoint", "frame_id": frame_identifier(frame),
                "unresolved_fields": {name: observed[name] for name in names
                                      if observed[name].get("accepted") is not True}}])
        fields.append(tuple(observed[name]["value"] for name in names))
        if (any(p.reason == 'inspected_title_same_hand' for p in predictions)
                and observed['turns_remaining']['value'] != context.gkh_hif_inspected_art[
                    'independent_detail']['after']['turns_remaining']):
            raise LiveSelectionError('inspected hand turn changed')
        if any(p.reason == 'retained_same_hand_family' for p in predictions):
            baseline_fields = context.gkh_hif_committed_event['core']['fields']
            if json.dumps(fields[-1]) != json.dumps(baseline_fields):
                raise LiveSelectionError('retained hand public state changed')
        if inventory_before is not None and families[-1] != inventory_before['families']:
            raise LiveSelectionError('inventory return visual family changed', events=[dict(
                phase='inventory_inspected_return_contradiction', frame_id=frame_identifier(frame),
                click_dispatched=False)])
        return_baseline = inventory_before if inventory_before is not None else return_observation
        if return_baseline is not None and fields[-1] != return_baseline['fields']:
            raise LiveSelectionError('inventory return Live resources changed', events=[dict(
                phase='inventory_inspected_return_contradiction', fields=fields[-1],
                frame_id=frame_identifier(frame), click_dispatched=False)])
    if unresolved_target:
        slot = return_observation['slot']
        expected = return_observation['families']
        independent = all(
            len(row) == len(expected) and all(
                p.slot == i and p.frame_id == frame_identifier(frame)
                and p.recognizer == 'card_art_embedding_top5_hif_masked_v1'
                and math.isfinite(p.confidence) and 0 <= p.confidence <= 1
                and math.isfinite(p.margin) and 0 <= p.margin <= 1
                and p.top_k_card_ids
                and ((p.accepted is True and p.reason == 'accepted_visual_family'
                      and p.confidence >= identity_threshold and p.margin >= identity_min_margin
                      and tuple(p.top_k_card_ids[0]) == expected[i])
                     or (i == slot and p.accepted is False and p.reason in (
                         'below_acceptance_threshold', 'below_margin_threshold')))
                for i, p in enumerate(row))
            for frame, row in zip(frames, return_predictions, strict=True))
        failure_sink = getattr(context, 'gkh_hif_inventory_failure_sink', None)
        if callable(failure_sink):
            for frame in frames:
                failure_sink(frame)
        raise LiveSelectionError('inventory inspected target remains unresolved', events=[dict(
            phase='inventory_inspected_target_transient' if independent else 'inventory_live_identity',
            slot=slot, frame_ids=[frame_identifier(frame) for frame in frames],
            families=families, fields=fields, click_dispatched=False,
            identity_observations=[[p.to_dict() for p in row] for row in return_predictions])])
    # A missing read can use a complete read of the identical title crop in
    # this checkpoint, provided the missed frame's own raw artwork also agrees.
    for index, failure, misses in pending_titles:
        row = list(families[index])
        for pixels, prediction in misses:
            proof = title_reads.get(pixels)
            if proof is None or not _hif_title_visual_match(
                    prediction, set(proof['family']), identity_min_margin)[1]:
                failure_sink = getattr(context, 'gkh_hif_inventory_failure_sink', None)
                if callable(failure_sink):
                    for failed_frame in frames:
                        failure_sink(failed_frame)
                raise LiveSelectionError('inventory navigation Live visual family is unresolved', events=[failure])
            slot = prediction.slot
            current_family_rows[index][slot] = row[slot] = proof['family']
            corrected_slots.add(slot)
            title_corroboration.append(dict(scope='current_title_same_pixels', accepted=True,
                slot=slot, family=proof['family'], frame_id=prediction.frame_id, roi=proof['roi'],
                source_frame_ids=(proof['frame_id'],), original_prediction=prediction.to_dict(),
                upgrade_verified=False))
        families[index] = tuple(row)
    title_slots = {proof['slot'] for proof in title_reads.values()}
    if (any(row.get(slot) != families[0][slot] for row in current_family_rows for slot in title_slots)
            or any(other is not None and i != slot and set(other).intersection(families[0][slot])
                   for slot in corrected_slots for i, other in enumerate(families[0]))):
        raise LiveSelectionError('current Live card title lacks consistent current slot evidence', events=[dict(
            phase='inventory_live_checkpoint_disagreement', frame_ids=frame_ids, families=families,
            title_family_corroboration=title_corroboration, click_dispatched=False)])
    return {"families": families[0], "fields": fields[0], "candidates": candidates[-1],
            "candidate_boxes": [[list(c.box) for c in row] for row in candidates],
            "frame_ids": frame_ids, "select_evidence": evidence,
            "title_family_corroboration": title_corroboration,
            "art_preprocessing_evidence": art_preprocessing_evidence}


def _inventory_inspected_return_scope(context, before):
    """Check the independent reference once, outside the return capture loop."""
    from live_decision.hif_inspected_art import independent_detail_reference_valid

    run = getattr(context, 'gkh_hif_live_context', None)
    receipt = getattr(context, 'gkh_hif_inspected_art', None)
    if (not isinstance(run, HifLiveRunContext) or not isinstance(receipt, dict)
            or (receipt.get('run_id'), receipt.get('revision')) != (run.run_id, run.revision)
            or not independent_detail_reference_valid(receipt)):
        return None
    slot = receipt['slot']
    families = tuple(tuple(family) for family in before['families'])
    fields = tuple(before['fields'])
    original = receipt['independent_detail']['after']
    if (not 0 <= slot < len(families) or families[slot] != tuple(receipt['family'])
            or families != tuple(tuple(family) for family in original['families'])
            or len(fields) != 6 or fields[0] != original['turns_remaining']):
        return None
    return dict(slot=slot, families=families, fields=fields, run_id=run.run_id, revision=run.revision)


def _observe_inventory_return(context, recognizer, before, event, events, identity_gate, *,
                              validated_scope=None, capture_frames=None, checkpoint=None,
                              observation_key='return_observations', retry_evidence=None,
                              observation_started_at=None):
    """Only a fully checked weak inspected target earns extra zero-input groups."""
    started = time.perf_counter()
    if observation_started_at is not None:
        if (type(observation_started_at) not in (int, float)
                or not math.isfinite(observation_started_at)
                or not 0 <= observation_started_at <= started):
            raise ValueError('initial observation start time is invalid')
        started = float(observation_started_at)
    scope = validated_scope if validated_scope is not None else _inventory_inspected_return_scope(context, before)
    limit_batch_time = scope is not None and getattr(context, 'gkh_hif_upstream_card_ui', False) is not True
    def read_checkpoint(frames):
        if len(frames) != 1:
            raise ValueError('HIF panel return requires the current frame')
        current = {'single_frame': True}
        return (checkpoint(frames, return_observation=scope, **current) if checkpoint is not None else
            _inventory_live_checkpoint(context, frames, recognizer, **identity_gate,
                **current,
                **({'return_observation': scope} if scope is not None else {})))

    for attempt in range(8 if scope is not None else 3):
        after_frames = ()
        try:
            if limit_batch_time and time.perf_counter()-started >= 10:
                raise LiveSelectionError('inventory return observation budget expired')
            after_frames = capture_frames() if capture_frames is not None else _capture_hif_panel_frames(context)
            after = read_checkpoint(after_frames)
            elapsed = (time.perf_counter()-started)*1000
            if limit_batch_time and elapsed >= 10000:
                raise LiveSelectionError('inventory return observation budget expired')
        except (LiveSelectionError, ValueError, RuntimeError) as error:
            detail = list(getattr(error, 'events', ()))
            elapsed = (time.perf_counter()-started)*1000
            transient = (scope is not None and len(detail) == 1
                and detail[0].get('phase') == 'inventory_inspected_target_transient'
                and detail[0].get('click_dispatched') is False
                and detail[0].get('frame_ids') == [frame_identifier(f) for f in after_frames])
            event.setdefault(observation_key, []).append(dict(
                accepted=False, reason=str(error), detail=detail,
                frame_ids=[frame_identifier(frame) for frame in after_frames],
                observation_attempt=attempt+1, elapsed_ms=elapsed))
            contradiction = any(row.get('phase') == 'inventory_inspected_return_contradiction' for row in detail)
            if (contradiction or attempt >= (7 if transient else 2)
                    or retry_evidence is not None and not transient and not retry_evidence(error, after_frames)
                    or limit_batch_time and elapsed >= 10000):
                raise LiveSelectionError(str(error), events=detail if events is None else events) from error
            time.sleep(.2)
            continue
        event.setdefault(observation_key, []).append(dict(accepted=True,
            frame_ids=after['frame_ids'], observation_attempt=attempt+1, elapsed_ms=elapsed))
        return after_frames, after


def _bind_inspected_complete_observer(context, checkpoint, scope, *, identity_threshold=.85, identity_min_margin=.05):
    """Scope an observation callback to this accepted core, never to inputs."""
    from live_decision.inventory import initial_checkpoint_retry_evidence

    run = getattr(context, 'gkh_hif_live_context', None)
    if (not isinstance(scope, dict) or not isinstance(run, HifLiveRunContext)
            or (scope.get('run_id'), scope.get('revision')) != (run.run_id, run.revision)):
        return

    def capture_complete(capture_frames, event):
        current = context.gkh_hif_live_context
        if (current.run_id, current.revision) != (scope['run_id'], scope['revision']):
            raise LiveSelectionError('complete observation belongs to another HIF runtime')
        return _observe_inventory_return(context, None, scope, event, None, {},
            validated_scope=scope, capture_frames=capture_frames, checkpoint=checkpoint,
            observation_key='inspected_live_observations', retry_evidence=initial_checkpoint_retry_evidence)

    def single_frame_reobserve_allowed(error, image, before):
        current = context.gkh_hif_live_context
        details = getattr(error, 'events', ())
        fid = frame_identifier(image)
        if ((current.run_id, current.revision) != (scope['run_id'], scope['revision'])
                or any(before.get(key) != scope[key] for key in ('families', 'fields'))
                or str(error) != 'inventory navigation Live visual family is unresolved'
                or len(details) != 1 or details[0].get('phase') != 'inventory_live_identity'
                or details[0].get('frame_id') != fid or details[0].get('frame_ids') != [fid]):
            return False
        predictions = details[0].get('predictions', ())
        if len(predictions) != len(scope['families']):
            return False
        for slot, prediction in enumerate(predictions):
            confidence, margin = prediction.get('confidence'), prediction.get('margin')
            if (prediction.get('slot') != slot
                    or prediction.get('recognizer') != 'card_art_embedding_top5_hif_masked_v1'
                    or type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1
                    or type(margin) not in (int, float) or not math.isfinite(margin) or not 0 <= margin <= 1
                    or not prediction.get('top_k_card_ids')):
                return False
            if slot == scope['slot']:
                if (prediction.get('accepted') is not False or prediction.get('reason') not in (
                        'below_acceptance_threshold', 'below_margin_threshold')):
                    return False
            elif (prediction.get('accepted') is not True or prediction.get('reason') != 'accepted_visual_family'
                    or confidence < identity_threshold or margin < identity_min_margin
                    or tuple(prediction['top_k_card_ids'][0]) != scope['families'][slot]):
                return False
        return True

    checkpoint.capture_complete = capture_complete
    checkpoint.single_frame_reobserve_allowed = single_frame_reobserve_allowed


def _observe_hif_inventory_transaction(context, recognizer, *, maximum_age_ms,
                                       identity_threshold, identity_min_margin, full_zones=False, items_overview=False, first_page_customizations=False, inspect_customizations=False):
    """Recognized overview navigation; optional inline details never play cards."""
    from dataclasses import asdict

    from utils.maa_input import verify_maa_input_endpoints
    from live_decision.inventory import (
        InventoryZoneCollector,
        InventoryThumbnailRecognizer,
        _inventory_ocr,
        _unique_inventory_text,
        observe_inventory_hand,
        read_inventory_sections,
        verify_inventory_heading,
        inventory_scroll_distance,
        recognize_inventory_button,
        inventory_scroll_displacement,
        capture_initial_live_checkpoint,
        fresh_identity_reobserve_allowed,
        inventory_identity_reobserve_allowed,
        inventory_known_count_reobserve_allowed,
    )
    from arena_winrate.badge_reference import BadgeReferenceGallery

    events = []
    limit_batch = getattr(context, 'gkh_hif_upstream_card_ui', False) is not True
    settlement_core = getattr(context, 'gkh_hif_settlement_cards_core', None)
    def capture_inventory_frames():
        return _capture_hif_panel_frames(context)
    if (not isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext)
            or type(maximum_age_ms) not in {int, float}
            or not math.isfinite(maximum_age_ms) or not 0 < maximum_age_ms <= 1000):
        raise LiveSelectionError("inventory transaction context or freshness gate is invalid")
    run = context.gkh_hif_live_context
    references = BadgeReferenceGallery.load(_resolve_runtime_path(
        Path("assets/resource/base/model/embedding/arena_badge_reference"),
        Path("resource/base/model/embedding/arena_badge_reference")))
    reader = InventoryThumbnailRecognizer(references, recognizer)
    customization_catalog = None
    retained_ambiguous = None
    partial_details = {}
    if full_zones and inspect_customizations and getattr(context, 'gkh_hif_opening_inventory_details', None):
        partial_details = context.gkh_hif_opening_inventory_details
        context.gkh_hif_opening_inventory_details = None
    confirmed_deck = (getattr(context, 'gkh_hif_confirmed_deck_baseline', None)
        if getattr(context, 'gkh_hif_permanent_inventory', None) is None else None)
    if first_page_customizations or inspect_customizations:
        from live_decision.inventory_customization import (
            PermanentCustomizationEvidenceRequired,
            load_customization_catalog,
            read_inventory_detail_text_rows,
            confirmed_inventory_detail_title,
            inspect_thumbnail_customizations,
        )
        customization_catalog = load_customization_catalog(CATALOG_PATH, _resolve_runtime_path(
            Path("assets/data/customizations.json"), Path("data/customizations.json")))
        retained = getattr(context, 'gkh_hif_permanent_inventory', None)
        if full_zones and retained is not None:
            from live_decision.inventory_customization import retained_ambiguous_card_ids
            if (retained[0].run_id, retained[0].revision) != (run.run_id, run.revision):
                raise LiveSelectionError('retained inventory observation epoch changed')
            retained_ambiguous = retained_ambiguous_card_ids(*retained)
        elif full_zones and confirmed_deck is not None:
            variants = {}
            for row in confirmed_deck['cards']:
                variants.setdefault(row['card_id'], set()).add(
                    tuple(sorted(row['customizations'].items())))
            retained_ambiguous = {card_id for card_id, groups in variants.items() if len(groups) > 1}
    identity_gate = {"identity_threshold": identity_threshold, "identity_min_margin": identity_min_margin}
    event = {"phase": "inventory_customization_observation" if first_page_customizations else "inventory_item_overview" if items_overview else "inventory_zone_observation" if full_zones else "inventory_hand_observation", "accepted": False,
             "input_click_count": 0}
    events.append(event)
    if settlement_core is not None and (
            items_overview or first_page_customizations
            or (settlement_core.get('run_id'), settlement_core.get('revision')) != (run.run_id, run.revision)
            or not isinstance(settlement_core.get('core'), dict)):
        raise LiveSelectionError('inventory settlement checkpoint context changed', events=events)

    def initial_checkpoint(frames):
        try:
            return _inventory_live_checkpoint(context, frames, recognizer, **identity_gate,
                **({'single_frame': True} if len(frames) == 1 else {}))
        except LiveSelectionError as error:
            failures = error.events
            # 1221/1225: the cards component can meet a new effect flash after
            # the settlement core passed. Each pre-navigation weak identity
            # read may consume the same existing three-batch budget.
            if (settlement_core is None or event['input_click_count'] != 0
                    or not frames
                    or str(error) != 'inventory navigation Live visual family is unresolved'
                    or not failures or any(f.get('phase') != 'inventory_live_identity'
                        or not f.get('predictions') or any(not p.get('top_k_card_ids')
                            or p.get('accepted') is not True and p.get('reason') not in (
                                'below_acceptance_threshold', 'below_margin_threshold')
                            for p in f['predictions']) for f in failures)):
                raise
            raise LiveSelectionError(str(error), events=[dict(
                phase='hif_settlement_hand_pending', reason=str(error),
                original_failure_events=failures,
                frame_ids=[frame_identifier(frame) for frame in frames])]) from error

    try:
        capture_before = getattr(context, 'gkh_hif_opening_cards_capture', capture_initial_live_checkpoint)
        before_frames, before = capture_before(
            capture_inventory_frames,
            initial_checkpoint, event)
        if settlement_core is not None:
            if any(json.dumps(before[name], sort_keys=True) !=
                   json.dumps(settlement_core['core'].get(name), sort_keys=True)
                   for name in ('families', 'fields')):
                raise LiveSelectionError('Live changed before settlement inventory navigation')
            if event.get('initial_observations'):
                event['settlement_initial_reobservation'] = dict(
                    frame_ids=before['frame_ids'], matched_initial_core=True, input_click_count=0)
    except ValueError as error:
        event['reason'] = str(error)
        event['checkpoint_failure_events'] = getattr(error, 'events', ())
        raise LiveSelectionError(str(error), events=events) from error
    event['before_frame_ids'] = before['frame_ids']
    if before.get('title_family_corroboration'):
        event['before_title_family_corroboration'] = before['title_family_corroboration']

    def click(box, captured_at, phase):
        try:
            endpoints = verify_maa_input_endpoints(context.tasker.controller.info, expected_client_size=getattr(context, "gkh_hif_client_size", (720, 1280)))
        except (RuntimeError, OSError, ValueError) as error:
            raise LiveSelectionError(str(error), events=events) from error
        if (time.perf_counter() - captured_at) * 1000 > maximum_age_ms:
            raise LiveSelectionError("inventory navigation snapshot expired", events=events)
        x, y, width, height = box
        # Count dispatches even if Maa reports failure: a failed result cannot
        # prove that Windows delivered no input.
        event["input_click_count"] += 1
        event.setdefault("inputs", []).append({"phase": phase, "box": tuple(box), "endpoints": endpoints})
        if not context.tasker.controller.post_click(x+width//2, y+height//2).wait().succeeded:
            raise LiveSelectionError("Maa inventory navigation click failed", events=events)
        time.sleep(SELECTION_SETTLE_SECONDS)

    def inspect_customization_page(frames, presence, collector):
        from live_decision.inventory_customization import unread_opening_candidates

        details = event.setdefault('detail_observations', [])
        resolved = {(row['zone'], row['index']) for row in details}
        candidates = [row for row in presence if row['presence'] == 'DETAIL_CANDIDATE'
                      and (row['zone'], row['index']) not in resolved]
        candidates = unread_opening_candidates(candidates, partial_details, details)
        if limit_batch and sum(not row.get('opening_detail_source') for row in details) + len(candidates) > 4:
            raise ValueError('inventory customization detail budget exceeded')
        for candidate in candidates:
            if limit_batch and time.perf_counter()-transaction_started > 55:
                raise ValueError('inventory customization transaction time budget exceeded')
            for freshness_attempt in range(3):
                fresh = context.tasker.controller.post_screencap().wait().get()
                captured_at = hif_captured_at(context.tasker.controller, fresh)
                rows = _inventory_ocr(context, fresh, 'LiveInventoryCustomizationPage')
                verify_inventory_heading(context, fresh, rows)
                _unique_inventory_text(rows, '閉じる', (210,1120,290,85))
                try:
                    # The page can coast after observation. As for
                    # pre-scroll rebinding, follow a uniquely measured shift;
                    # the old pixel position never authorizes the detail click.
                    displacement = inventory_scroll_displacement(frames[len(frames)//2], fresh,
                        collector.previous_boxes, minimum=-2, maximum=80)
                    break
                except ValueError as error:
                    event.setdefault('customization_alignment_rejections', []).append(dict(
                        reason=str(error), frame_id=frame_identifier(fresh)))
                    sink = getattr(context, 'gkh_hif_alignment_failure_sink', None)
                    if callable(sink):
                        sink(frames[len(frames)//2], (fresh,), collector.previous_boxes)
                    if freshness_attempt == 2 or (limit_batch and time.perf_counter()-transaction_started > 55):
                        raise
                    time.sleep(.2)
            x, y, width, height = candidate['box']
            current_box = (x, y-displacement, width, height)
            if (current_box[1] < 406 or current_box[1]+height > 1007
                    or x+width > 606 and current_box[1]+height > 947):
                raise ValueError('customization thumbnail moved outside the visible page')
            current_identity = reader.identify(fresh, current_box, plan=_infer_plan(context))
            event.setdefault('customization_preselection', []).append(dict(
                card_id=candidate['card_id'], source_box=list(candidate['box']),
                current_box=list(current_box), displacement=displacement,
                observed_card_id=current_identity.card_id, frame_id=frame_identifier(fresh)))
            if current_identity.card_id != candidate['card_id']:
                raise ValueError('customization thumbnail identity changed before selection')
            # The current indexed slot and freshly confirmed artwork
            # authorize this page-local target, including scrolled pages.
            click(current_box, captured_at, 'customization_select')
            permanent_bundle = getattr(context, 'gkh_hif_opening_state', None)
            same_bundle = (permanent_bundle is not None
                and (permanent_bundle.run_id, permanent_bundle.revision) == (run.run_id, run.revision))
            for settling in range(3):
                detail_frames = capture_inventory_frames()
                observations = [read_inventory_detail_text_rows(context, frame) for frame in detail_frames]
                try:
                    from live_decision.inventory_support import resolve_inventory_detail
                    from live_decision.inventory_detail_badge import read_inventory_detail_badge_count

                    detail_frame_ids = [frame_identifier(frame) for frame in detail_frames]
                    detail_observation = resolve_inventory_detail(customization_catalog,
                        candidate, observations, detail_frame_ids,
                        bundle=permanent_bundle if same_bundle else None,
                        badge_observer=lambda: read_inventory_detail_badge_count(
                            context, detail_frames, detail_frame_ids))
                    break
                except ValueError as error:
                    event.setdefault('detail_rejections', []).append({
                        'card_id':candidate['card_id'], 'reason':str(error),
                        'rows':[[asdict(row) if hasattr(row,'__dataclass_fields__') else
                            {'text':row.text,'score':row.score,'box':row.box} for row in rows]
                            for rows in observations],
                        'frame_ids':[frame_identifier(frame) for frame in detail_frames]})
                    if isinstance(error, PermanentCustomizationEvidenceRequired) or settling == 2:
                        failure_sink = getattr(context, 'gkh_hif_inventory_failure_sink', None)
                        if callable(failure_sink):
                            for frame in detail_frames:
                                failure_sink(frame)
                        raise
                    time.sleep(.2)
            event['detail_observations'].append(detail_observation)
            collector.selected_card_title = confirmed_inventory_detail_title(
                customization_catalog, candidate['card_id'], observations)

    # Read the button on a new Live frame after the reference and hand reads.
    fresh_times = []
    def capture_one_live_frame():
        frame = context.tasker.controller.post_screencap().wait().get()
        fresh_times.append(hif_captured_at(context.tasker.controller, frame))
        return (frame,)
    for entry_attempt in range(3):
        try:
            (fresh,), current = capture_initial_live_checkpoint(capture_one_live_frame,
                lambda frames: _inventory_live_checkpoint(context, frames, recognizer,
                    single_frame=True, **identity_gate),
                event.setdefault('preinput_resource_checkpoint', {}))
            if current.get('title_family_corroboration'):
                event['preinput_resource_checkpoint']['title_family_corroboration'] = current['title_family_corroboration']
        except ValueError as error:
            if entry_attempt < 2 and (str(error) == 'inventory fresh recheck is not an unselected hand'
                    or fresh_identity_reobserve_allowed(error, before)):
                _, confirmed = capture_initial_live_checkpoint(capture_inventory_frames, initial_checkpoint, event)
                if (confirmed['families'], confirmed['fields']) != (before['families'], before['fields']):
                    raise LiveSelectionError('Live changed before opening inventory', events=events) from error
                event.setdefault('preinput_hand_reobservations', []).append(confirmed['frame_ids'])
                continue
            event['checkpoint_failure_events'] = getattr(error, 'events', ())
            event['reason'] = str(error)
            raise LiveSelectionError(str(error), events=events) from error
        captured_at = fresh_times[-1]
        if (current["families"], current["fields"]) != (before["families"], before["fields"]):
            raise LiveSelectionError("Live changed before opening inventory", events=events)
        targets, entry_image, entry_templates, enlarged = recognize_inventory_button(
            context, fresh, getattr(context, "gkh_hif_client_size", (720, 1280)))
        if enlarged:
            event.setdefault('entry_raster_fallbacks', []).append(dict(
                frame_id=frame_identifier(fresh), templates=entry_templates,
                reason='primary_zero_candidates_same_frame_enlarged_raster', input_click_count=0))
        if len(targets) == 1:
            break
        event.setdefault("entry_reobservations", []).append(dict(frame_id=frame_identifier(fresh), reason="Live inventory button is not unique"))
        if entry_attempt == 2:
            event['entry_failure_pixels'] = dict(source_frame_id=frame_identifier(fresh),
                processed_frame_id=frame_identifier(entry_image), templates=entry_templates)
            failure_sink = getattr(context, 'gkh_hif_inventory_failure_sink', None)
            if callable(failure_sink):
                failure_sink(fresh)
                if frame_identifier(entry_image) != frame_identifier(fresh):
                    failure_sink(entry_image)
            raise LiveSelectionError("Live inventory button is not unique", events=events)
        time.sleep(.08)
    click(tuple(targets[0].box), captured_at, "open")
    failure = None
    inventory = None
    transaction_started = time.perf_counter()
    try:
        # Opening an overlay is a page transition, not a card-selection
        # transition. Await its anchors without dispatching another click.
        for attempt in range(3):
            arrival = context.tasker.controller.post_screencap().wait().get()
            rows = _inventory_ocr(context, arrival, "LiveInventoryArrival")
            try:
                verify_inventory_heading(context, arrival, rows)
                _unique_inventory_text(rows, "閉じる", (210, 1120, 290, 85))
            except ValueError as error:
                event.setdefault("page_arrival_observations", []).append({
                    "frame_id": frame_identifier(arrival), "accepted": False, "reason": str(error)})
                if attempt == 2:
                    raise
                time.sleep(.2)
                continue
            event.setdefault("page_arrival_observations", []).append({
                "frame_id": frame_identifier(arrival), "accepted": True})
            break
        if items_overview:
            fresh = context.tasker.controller.post_screencap().wait().get()
            captured_at = hif_captured_at(context.tasker.controller, fresh)
            rows = _inventory_ocr(context,fresh,"LiveInventoryItemTab")
            _unique_inventory_text(rows,"所持スキルカード",(15,25,285,65))
            _unique_inventory_text(rows,"閉じる",(210,1120,290,85))
            tab = _unique_inventory_text(rows,"Pアイテム",(450,1010,170,70))
            click(tuple(tab.box),captured_at,"items_tab")
            from live_decision.item_panel import read_item_panel, item_panel_consensus, item_panel_stable_geometry
            from live_decision.status_panel import join_status_panel_pages, _join_status_panel_prefix
            item_pages=[]
            event['item_pages']=[]
            while True:
                if limit_batch and (len(item_pages)>=14 or time.perf_counter()-transaction_started>55):
                    raise ValueError('item overview navigation budget exceeded')
                thumb_height=(item_pages[0].thumb[1]-item_pages[0].thumb[0]) if item_pages else None
                for settling in range(8):
                    if limit_batch and time.perf_counter()-transaction_started>55:
                        raise ValueError('item overview settling time budget exceeded')
                    item_frames=capture_inventory_frames()
                    event['item_overview_frame_ids']=[frame_identifier(f) for f in item_frames]
                    try:
                        if len(item_frames) > 1:
                            item_panel_stable_geometry(item_frames,expected_thumb_height=thumb_height)
                        page=item_panel_consensus([read_item_panel(context,f,expected_thumb_height=thumb_height) for f in item_frames])
                        _join_status_panel_prefix((*item_pages,page))
                        break
                    except ValueError as error:
                        event.setdefault('item_page_rejections',[]).append({
                            'reason':str(error),'frame_ids':event['item_overview_frame_ids']})
                        if settling==7 or 'settled' not in str(error):
                            raise
                        time.sleep(.2)
                item_pages.append(page)
                event['item_pages'].append({'page':asdict(page),'frame_ids':event['item_overview_frame_ids']})
                if page.at_bottom:
                    break
                fresh=context.tasker.controller.post_screencap().wait().get()
                captured_at = hif_captured_at(context.tasker.controller, fresh)
                current_page=read_item_panel(context,fresh,expected_thumb_height=page.thumb[1]-page.thumb[0])
                if (current_page.content_key!=page.content_key
                        or max(abs(a-b) for a,b in zip(current_page.thumb,page.thumb))>2):
                    raise ValueError('item overview changed before scrolling')
                endpoints=verify_maa_input_endpoints(context.tasker.controller.info, expected_client_size=getattr(context, "gkh_hif_client_size", (720, 1280)))
                if (time.perf_counter()-captured_at)*1000>maximum_age_ms:
                    raise ValueError('item overview scroll snapshot expired')
                event['input_click_count']+=1
                event.setdefault('inputs',[]).append({'phase':'items_scroll','endpoints':endpoints,
                                                     'begin':(400,850),'end':(400,770)})
                if not context.tasker.controller.post_swipe(400,850,400,770,400).wait().succeeded:
                    raise ValueError('Maa item overview scroll failed')
                time.sleep(.35)
            item_rows=join_status_panel_pages(item_pages)
            event['item_rows']=[asdict(row) for row in item_rows]
            event['complete_item_list_observed']=True
            event['item_identity_state_bound']=False
            inventory={'accepted':True,'card_ids':()}
        elif first_page_customizations:
            collector = InventoryZoneCollector(len(before["families"]), live_hand_evidence=dict(
                source='inventory_navigation_before', run_id=run.run_id, revision=run.revision,
                frame_ids=before['frame_ids'], families=before['families']))
            # As with full-zone reading, visible overlay anchors do not prove
            # that the card headers have finished their entrance animation.
            # Read the new page again if its heading is still moving.
            for settling in range(8):
                if limit_batch and time.perf_counter()-transaction_started > 55:
                    raise ValueError('inventory customization settling time budget exceeded')
                frames = capture_inventory_frames()
                try:
                    page = collector.observe(context, frames, reader, plan=_infer_plan(context))
                    break
                except ValueError as error:
                    event.setdefault('page_rejections', []).append({
                        'reason':str(error), 'frame_ids':[frame_identifier(f) for f in frames]})
                    if settling == 7 or str(error) not in (
                            'inventory header positions have not settled',
                            'inventory section headers have not settled'):
                        raise
                    time.sleep(.2)
            presence = inspect_thumbnail_customizations(frames, page['slots'])
            event['first_page_presence'] = presence
            inspect_customization_page(frames, presence, collector)
            event['scope'] = 'first_visible_page_only'
            inventory = {'accepted':True, 'card_ids':tuple(
                row[2] for row in page['slots'] if row[0] == '手札')}
        elif full_zones:
            collector = InventoryZoneCollector(len(before["families"]), live_hand_evidence=dict(
                source='inventory_navigation_before', run_id=run.run_id, revision=run.revision,
                frame_ids=before['frame_ids'], families=before['families']))
            event["pages"] = []
            badge_by_slot = {}
            page_refresh = False
            page_refresh_count = 0
            swipe_pending = False
            ineffective_swipes = 0
            while not collector.complete:
                identity_reobservations = 0
                previous_displacement = collector.displacement
                if limit_batch and time.perf_counter()-transaction_started > 55:
                    raise ValueError("inventory zone transaction time budget exceeded")
                for settling in range(8):
                    if limit_batch and time.perf_counter()-transaction_started > 55:
                        raise ValueError("inventory zone settling time budget exceeded")
                    inventory_frames = capture_inventory_frames()
                    try:
                        page = collector.observe(context,inventory_frames,reader,
                            plan=_infer_plan(context),recheck=page_refresh)
                        page_refresh = False
                        break
                    except ValueError as error:
                        event.setdefault("page_rejections",[]).append({
                            "reason":str(error),"detail":getattr(error,'evidence',None),
                            "frame_ids":[frame_identifier(f) for f in inventory_frames]})
                        failure_sink = getattr(context,'gkh_hif_inventory_failure_sink',None)
                        if callable(failure_sink) and str(error)=='inventory page contains an unresolved card':
                            for frame in inventory_frames:
                                failure_sink(frame)
                        sink = getattr(context,'gkh_hif_alignment_failure_sink',None)
                        if callable(sink) and 'translation' in str(error) and collector.previous is not None:
                            sink(collector.previous, inventory_frames, collector.previous_boxes)
                        identity_retry = (inventory_identity_reobserve_allowed(error)
                            or inventory_known_count_reobserve_allowed(error,collector.sections)) and identity_reobservations < 2
                        if identity_retry:
                            identity_reobservations += 1
                        if settling == 7 or not (identity_retry or any(term in str(error) for term in ("settled","translation"))):
                            raise
                        time.sleep(.2)
                if inspect_customizations:
                    from live_decision.inventory_customization import inspect_thumbnail_customizations
                    badge_slots = page['slots'] if retained_ambiguous is None else [
                        slot for slot in page['slots'] if slot[2] in retained_ambiguous]
                    observed_badges = inspect_thumbnail_customizations(inventory_frames, badge_slots)
                    badge_frame_ids = [frame_identifier(frame) for frame in inventory_frames]
                    for badge in observed_badges:
                        key = (badge['zone'], badge['index'])
                        compact = {name:badge[name] for name in ('zone','index','card_id','presence','reason')}
                        compact['frame_ids'] = badge_frame_ids
                        if confirmed_deck is not None:
                            compact.update(box=list(badge['box']), observations=badge['observations'])
                        if key in badge_by_slot and any(
                            badge_by_slot[key][field] != compact[field] for field in ('card_id','presence')
                        ):
                            event['customization_overlap_conflict'] = {
                                'previous': badge_by_slot[key], 'current': compact,
                                'box': list(badge['box']), 'observations': badge['observations'],
                                'frame_ids': [frame_identifier(f) for f in inventory_frames],
                            }
                            failure_sink = getattr(context, 'gkh_hif_inventory_failure_sink', None)
                            if callable(failure_sink):
                                for frame in inventory_frames:
                                    failure_sink(frame)
                            raise ValueError('inventory overlapping customization evidence changed')
                        badge_by_slot[key] = compact
                    event['customization_presence'] = list(badge_by_slot.values())
                    if confirmed_deck is None:
                        inspect_customization_page(inventory_frames, observed_badges, collector)
                event["pages"].append({**page,"frame_ids":[frame_identifier(f) for f in inventory_frames]})
                if collector.complete:
                    break
                if swipe_pending:
                    movement = collector.displacement - previous_displacement
                    ineffective_swipes = ineffective_swipes + 1 if movement < 4 else 0
                    event.setdefault('scroll_effects', []).append({
                        'translation': movement,
                        'state': 'negligible' if movement < 4 else 'effective',
                        'consecutive_ineffective': ineffective_swipes})
                    swipe_pending = False
                    if ineffective_swipes >= 2:
                        raise ValueError('inventory swipe remained ineffective after one confirmed compensation')
                if limit_batch and (collector.page_count >= 12 or time.perf_counter()-transaction_started > 55):
                    raise ValueError("inventory zone navigation budget exceeded")
                # A short scroll is authorized by fresh page identity and by
                # the same complete opaque card patches used for registration.
                for freshness_attempt in range(3):
                    fresh = context.tasker.controller.post_screencap().wait().get()
                    captured_at = hif_captured_at(context.tasker.controller, fresh)
                    needs_header_anchors = len({box[0] for box in collector.previous_boxes.values()}) == 1
                    total,fresh_sections = read_inventory_sections(context,fresh,
                        headers_required=needs_header_anchors,
                        require_unselected_panel=getattr(collector,'selected_card_title',None) is None)
                    if total != collector.total:
                        raise ValueError("inventory count changed before scrolling")
                    try:
                        adjustment = inventory_scroll_displacement(collector.previous,fresh,
                            collector.previous_boxes,minimum=-2,maximum=2,
                            reference_sections=getattr(collector,'previous_sections',()),
                            current_sections=fresh_sections)
                        break
                    except ValueError as error:
                        event.setdefault('pre_scroll_alignment_rejections',[]).append({
                            'reference_frame_id':frame_identifier(collector.previous),
                            'fresh_frame_id':frame_identifier(fresh),'reason':str(error)})
                        sink = getattr(context,'gkh_hif_alignment_failure_sink',None)
                        if callable(sink):
                            sink(collector.previous,(fresh,),collector.previous_boxes)
                        if freshness_attempt == 2 or (limit_batch and time.perf_counter()-transaction_started > 55):
                            if page_refresh_count < 2 and (not limit_batch or time.perf_counter()-transaction_started <= 55):
                                try:
                                    movement = inventory_scroll_displacement(collector.previous,fresh,
                                        collector.previous_boxes,minimum=3,maximum=80,
                                        reference_sections=getattr(collector,'previous_sections',()),
                                        current_sections=fresh_sections)
                                except ValueError:
                                    raise ValueError("inventory page moved before scrolling") from error
                                page_refresh_count += 1
                                page_refresh = True
                                event.setdefault('coasting_page_refreshes',[]).append({
                                    'translation':movement,'reference_frame_id':frame_identifier(collector.previous),
                                    'fresh_frame_id':frame_identifier(fresh)})
                                break
                            raise ValueError("inventory page moved before scrolling") from error
                        # A single rendered frame cannot invalidate a settled
                        # page. A new frame must still match its fixed anchors;
                        # do not move the reference or dispatch another swipe.
                else:
                    raise ValueError('inventory fresh checkpoint did not finish')
                if page_refresh:
                    # Re-read every visible identity/header in three frames;
                    # the measured movement alone does not authorize a swipe.
                    continue
                event.setdefault('pre_scroll_alignment',[]).append({'translation':adjustment})
                endpoints = verify_maa_input_endpoints(context.tasker.controller.info, expected_client_size=getattr(context, "gkh_hif_client_size", (720, 1280)))
                if (time.perf_counter()-captured_at)*1000 > maximum_age_ms:
                    raise ValueError("inventory scroll snapshot expired")
                event['input_click_count'] += 1
                scroll_distance = inventory_scroll_distance(collector.previous_boxes)
                event.setdefault('inputs',[]).append({'phase':'scroll','endpoints':endpoints,
                                                     'begin':(400,850),'end':(400,850-scroll_distance),
                                                     'compensating': ineffective_swipes == 1})
                if not context.tasker.controller.post_swipe(400,850,400,850-scroll_distance,400).wait().succeeded:
                    raise ValueError("Maa inventory scroll failed")
                swipe_pending = True
                time.sleep(.35)
            inventory = {'accepted':True,'card_ids':collector.composition()['手札'],
                         'composition':collector.composition()}
            event['composition'] = inventory['composition']
            event['complete_zone_counts'] = {name:len(ids) for name,ids in inventory['composition'].items()}
        else:
            inventory_frames = capture_inventory_frames()
            inventory = observe_inventory_hand(context, inventory_frames, reader,
                plan=_infer_plan(context), expected_count=len(before["families"]))
            event["inventory_frame_ids"] = tuple(frame_identifier(frame) for frame in inventory_frames)
            event["inventory_card_ids"] = inventory["card_ids"]
            event["inventory_observations"] = [
                {"box": row["box"], "card_id": row["card_id"],
                 "observations": [asdict(observation) for observation in row["observations"]]}
                for row in inventory["identities"]]
    except (ValueError, RuntimeError) as error:
        failure = str(error)
        event["read_failure"] = failure
    # Cleanup uses fresh, unique page anchors even when hand recognition failed.
    # Unknown pages are left untouched; there is no coordinate fallback.
    close_frame = context.tasker.controller.post_screencap().wait().get()
    captured_at = hif_captured_at(context.tasker.controller, close_frame)
    try:
        rows = _inventory_ocr(context, close_frame, "LiveInventoryClosePage")
        if items_overview:
            titles = []
            for title in ("所持スキルカード","所持Pアイテム"):
                try:
                    titles.append(_unique_inventory_text(rows,title,(15,25,285,65)))
                except ValueError:
                    pass
            if len(titles)!=1:
                raise ValueError("inventory item close page is ambiguous")
        else:
            verify_inventory_heading(context, close_frame, rows)
        close = _unique_inventory_text(rows, "閉じる", (210, 1120, 290, 85))
    except ValueError as error:
        raise LiveSelectionError("inventory close page is unresolved: " + str(error), events=events) from error
    click(tuple(close.box), captured_at, "close")
    return_gate = (dict(identity_gate, inventory_before=before)
        if getattr(context, 'gkh_hif_upstream_card_ui', False) is True else identity_gate)
    after_frames, after = _observe_inventory_return(context, recognizer, before, event, events, return_gate)
    event["after_frame_ids"] = after["frame_ids"]
    if after.get('title_family_corroboration'):
        event['after_title_family_corroboration'] = after['title_family_corroboration']
    event["wall_ms"] = (time.perf_counter()-transaction_started)*1000
    event["live_state_unchanged"] = (after["families"], after["fields"]) == (before["families"], before["fields"])
    if failure or not event["live_state_unchanged"] or not inventory or not inventory["accepted"]:
        event["reason"] = failure or "inventory identity or return state unresolved"
        raise LiveSelectionError(event["reason"], events=events)
    event["accepted"] = True
    return inventory["card_ids"], after_frames, after, events


def _carry_hif_selected_neighbor(context, prediction, *, evidence,
        identity_threshold, identity_min_margin, journal=None):
    from live_decision.hif_gray_selection import carry_selected_right_neighbor_identity

    proof = carry_selected_right_neighbor_identity(
        context.gkh_hif_opening_state,
        complete_event=context.gkh_hif_committed_event,
        journal=(journal if journal is not None else context.gkh_hif_gray_selected_source),
        prediction=prediction, evidence=evidence,
        identity_threshold=identity_threshold, identity_min_margin=identity_min_margin,
        geometry_comparator=_same_live_candidate_geometry)
    raw = dict(proof['carried_prediction'])
    for key in ('box', 'source_frames', 'top_k_scores'):
        raw[key] = tuple(raw[key])
    raw['top_k_card_ids'] = tuple(tuple(family) for family in raw['top_k_card_ids'])
    return replace(prediction, **raw), proof


def _existing_unselected_identity_source(context, frames, rows, raw, resolved, *,
                                         recognizer, plan, threshold, margin, evidence):
    """Choose one whole identity source from already verified unselected frames."""
    baseline = getattr(context, 'gkh_hif_committed_event', None)
    if (not isinstance(baseline, dict) or baseline.get('strategy_state_bound') is not True
            or len(frames) != 3 or len(rows) != 3
            or len({frame_identifier(frame) for frame in frames}) != 3):
        return None
    try:
        scope = _inventory_inspected_return_scope(context, baseline['core'])
        if scope is None:
            return None
        _retained_hif_hand_ids(context, scope['families'])
    except (KeyError, ValueError):
        return None
    slot, families = scope['slot'], scope['families']

    def compatible(predictions, frame, candidates):
        return len(predictions) == len(candidates) == len(families) and all(
            p.slot == i == candidates[i].slot and tuple(p.box) == tuple(candidates[i].box)
            and p.frame_id == frame_identifier(frame)
            and p.recognizer == 'card_art_embedding_top5_hif_masked_v1'
            and math.isfinite(p.confidence) and 0 <= p.confidence <= 1
            and math.isfinite(p.margin) and 0 <= p.margin <= 1 and p.top_k_card_ids
            and ((p.accepted is True and p.reason == 'accepted_visual_family'
                  and p.confidence >= threshold and p.margin >= margin
                  and tuple(p.top_k_card_ids[0]) == families[i])
                 or (i == slot and p.accepted is False
                     and p.reason in ('below_acceptance_threshold', 'below_margin_threshold')))
            for i, p in enumerate(predictions))

    if ([p.slot for p in resolved if not p.accepted] != [slot]
            or not compatible(raw, frames[0], rows[0])):
        return None
    from live_decision.hif_inspected_art import corroborate

    attempts = [dict(frame_id=frame_identifier(frames[0]), accepted=False,
        predictions=[p.to_dict() for p in raw], reason='inspected_target_transient')]
    evidence['identity_source_alternatives'] = attempts
    for index in (1, 2):
        frame, candidates = frames[index], rows[index]
        predictions = classify_live_card_views(recognizer, frame, candidates,
            frame_id=frame_identifier(frame), plan=plan, acceptance_threshold=threshold,
            min_margin=margin, **_live_card_art_profile(context, candidates, None))
        attempt = dict(frame_id=frame_identifier(frame), accepted=False,
            predictions=[p.to_dict() for p in predictions])
        attempts.append(attempt)
        if not compatible(predictions, frame, candidates):
            attempt['reason'] = 'identity_source_contradiction_or_other_weak_slot'
            raise LiveSelectionError('remaining Live identity source is contradictory', events=[evidence])
        proofs = []
        predictions = corroborate(context, frame, candidates, predictions,
            selected_slot=None, evidence=proofs)
        if not all(p.accepted and tuple(p.top_k_card_ids[0]) == families[i]
                   for i, p in enumerate(predictions)):
            attempt['reason'] = 'inspected_target_transient'
            continue
        attempt.update(accepted=True, reason='same_batch_whole_hand_independently_accepted')
        evidence.pop('bound_complete_title_families', None)
        evidence.pop('inspected_title_art_families', None)
        if proofs:
            evidence['inspected_title_art_families'] = proofs
        return frame, candidates, list(predictions)
    return None


def _observe_initial_identity_window(context, frames, *, recognizer, threshold, margin,
                                     evidence, observation_started_at):
    """Replace an all-weak initial group only with a complete new observation."""
    if observation_started_at is not None and (
            type(observation_started_at) not in (int, float) or not math.isfinite(observation_started_at)
            or not 0 <= observation_started_at <= time.perf_counter()):
        raise ValueError('initial observation start time is invalid')
    limit_batch_time = getattr(context, 'gkh_hif_upstream_card_ui', False) is not True
    attempts = evidence.get('identity_source_alternatives', ())
    baseline = getattr(context, 'gkh_hif_committed_event', None)
    if (not attempts and observation_started_at is not None
            and isinstance(baseline, dict) and baseline.get('strategy_state_bound') is True
            and baseline.get('accepted') is True and len(frames) == 1
            and len({frame_identifier(frame) for frame in frames}) == len(frames)):
        # A newly drawn animated card is not necessarily an inspected-return
        # target. Each retry must establish the whole hand and public state.
        before = baseline['core']
        _retained_hif_hand_ids(context, tuple(tuple(row) for row in before['families']))
        observed_frames = frames
        for attempt in range(3):
            if limit_batch_time and time.perf_counter()-observation_started_at >= 10:
                raise LiveSelectionError('initial identity observation budget expired', events=(evidence,))
            if attempt:
                observed_frames = _capture_hif_panel_frames(context)
            try:
                core = _inventory_live_checkpoint(context, observed_frames, recognizer,
                    identity_threshold=threshold, identity_min_margin=margin,
                    **({'single_frame': True} if len(observed_frames) == 1 else {}))
            except LiveSelectionError as error:
                evidence.setdefault('identity_window_observations', []).append(dict(
                    accepted=False, reason=str(error), observation_attempt=attempt+1,
                    frame_ids=[frame_identifier(f) for f in observed_frames]))
                if str(error) != 'inventory navigation Live visual family is unresolved' or attempt == 2:
                    raise
                continue
            _retained_hif_hand_ids(context, core['families'])
            if tuple(core['fields']) != tuple(before['fields']):
                raise LiveSelectionError('initial identity observation public state changed')
            if limit_batch_time and time.perf_counter()-observation_started_at >= 10:
                raise LiveSelectionError('initial identity observation budget expired')
            evidence.setdefault('identity_window_observations', []).append(dict(
                accepted=True, observation_attempt=attempt+1, frame_ids=core['frame_ids']))
            return observed_frames, core
    if (observation_started_at is None or len(attempts) != 1
            or [row.get('frame_id') for row in attempts] != [frame_identifier(f) for f in frames]
            or any(row.get('accepted') is not False or row.get('reason') != 'inspected_target_transient'
                   for row in attempts)
            or not isinstance(baseline, dict) or baseline.get('strategy_state_bound') is not True):
        return None
    scope = _inventory_inspected_return_scope(context, baseline['core'])
    if scope is None:
        return None
    _retained_hif_hand_ids(context, scope['families'])
    first = True

    def capture():
        nonlocal first
        if first:
            first = False
            return frames
        return _capture_hif_panel_frames(context)

    return _observe_inventory_return(context, recognizer, baseline['core'], evidence, None,
        dict(identity_threshold=threshold, identity_min_margin=margin), validated_scope=scope,
        capture_frames=capture, observation_key='identity_window_observations',
        retry_evidence=lambda error, observed: False,
        observation_started_at=observation_started_at)


def _observe_cards(
    context: Context,
    image: Any,
    frame_id: str,
    upgrade_images: tuple[Any, Any, Any],
    *,
    params: dict[str, Any],
    turns_remaining: int,
    input_deadline: float | None = None,
    recognizer: Any | None = None,
    catalog: SkillCardFeatureCatalog | None = None,
    selected_resource_baseline: dict[str, Any] | None = None,
    identity_observation_started_at: float | None = None,
) -> tuple[list[dict[str, Any]], int | None, int, list[dict[str, Any]]]:
    identity_threshold = _strict_finite_number(
        params,
        "debug_identity_threshold",
        default=0.70,
        lower=0.0,
        upper=1.0,
    )
    identity_min_margin = _strict_finite_number(
        params,
        "debug_identity_min_margin",
        default=0.0,
        lower=0.0,
        upper=1.0,
    )
    if identity_threshold is None or identity_min_margin is None:
        raise ValueError("identity thresholds must be approved numbers")
    if recognizer is None:
        recognizer = EmbeddingCardRecognizer.load(MODEL_ROOT)
    if catalog is None:
        catalog = SkillCardFeatureCatalog.load(CATALOG_PATH)
    plan = _infer_plan(context)

    live_mode = params.get("live_decision_mode")
    expected_initial_hand_count = (
        params.get("debug_expected_initial_hand_count")
        if live_mode in {"shadow_transaction", "debug_execute", "opening_shadow_transaction", "opening_execute"}
        and type(params.get("debug_expected_initial_hand_count")) is int
        and params.get("debug_expected_initial_hand_count") == 3
        else None
    )
    initial_fragment_reconciliations: list[dict[str, Any]] = []
    (
        initial_selection,
        initial_frame_candidates,
        initial_select_evidence,
        initial_geometry_stable,
    ) = _observe_live_selected_slot_consensus(
        context,
        upgrade_images,
        expected_selected_slot=None,
        reconciliation_events=initial_fragment_reconciliations,
        expected_hand_count=expected_initial_hand_count,
    )
    selection_events: list[dict[str, Any]] = [
        {
            "phase": "initial_observation",
            "attempted_slot": None,
            "click_dispatched": False,
            "frame_ids": [frame_identifier(frame) for frame in upgrade_images],
            "candidate_boxes": [
                [list(candidate.box) for candidate in observed]
                for observed in initial_frame_candidates
            ],
            "geometry_stable": initial_geometry_stable,
            "select_consensus": (
                initial_selection.to_dict() if initial_selection is not None else None
            ),
            "select_evidence": initial_select_evidence,
            "candidate_fragment_reconciliations": initial_fragment_reconciliations,
        }
    ]
    if not initial_geometry_stable or initial_selection is None:
        raise LiveSelectionError(
            "initial three-frame Live card geometry is unstable",
            events=selection_events,
        )
    # A settled HIF hand can still be animating in the first capture. When the
    # final two already-observed rows agree, classify that current whole row;
    # do not force a narrow first-frame crop onto the settled hand.
    identity_source_index = 0
    if (isinstance(getattr(context, 'gkh_hif_live_context', None), HifLiveRunContext)
            and isinstance(getattr(context, 'gkh_hif_committed_event', None), dict)
            and context.gkh_hif_committed_event.get('strategy_state_bound') is True
            and initial_selection.accepted and initial_selection.selected_slot is None
            and len(initial_frame_candidates[-1]) ==
                len(context.gkh_hif_committed_event.get('core', {}).get('families', ()))
            and tuple(c.box for c in initial_frame_candidates[-2])
                == tuple(c.box for c in initial_frame_candidates[-1])):
        identity_source_index = 2
    identity_source_image = upgrade_images[identity_source_index]
    identity_source_frame_id = frame_identifier(identity_source_image)
    identity_source_candidates = initial_frame_candidates[identity_source_index]
    identity_frame_candidates = initial_frame_candidates
    identity_source_event = selection_events[0]
    identity_source_event['identity_source_frame_index'] = identity_source_index
    identity_source_kind = "initial"
    for retry_index in range(2):
        selected_slots = {
            slot
            for observation in initial_selection.observations
            for slot in observation
        }
        retryable_missing_frame = bool(
            live_mode in {"selection_observe", "debug_execute"}
            and not initial_selection.accepted
            and initial_selection.reason == "select_frames_disagree"
            and len(selected_slots) == 1
        )
        if not retryable_missing_frame:
            break
        time.sleep(SELECTION_SETTLE_SECONDS)
        retry_images = _capture_three_frames(context)
        retry_fragment_reconciliations: list[dict[str, Any]] = []
        (
            retry_selection,
            retry_frame_candidates,
            retry_select_evidence,
            retry_geometry_stable,
        ) = _observe_live_selected_slot_consensus(
            context,
            retry_images,
            expected_selected_slot=None,
            reconciliation_events=retry_fragment_reconciliations,
            expected_hand_count=expected_initial_hand_count,
        )
        selection_events.append(
            {
                "phase": "initial_observation_retry",
                "retry_index": retry_index + 1,
                "attempted_slot": None,
                "click_dispatched": False,
                "frame_ids": [frame_identifier(frame) for frame in retry_images],
                "candidate_boxes": [
                    [list(candidate.box) for candidate in observed]
                    for observed in retry_frame_candidates
                ],
                "geometry_stable": retry_geometry_stable,
                "select_consensus": (
                    retry_selection.to_dict() if retry_selection is not None else None
                ),
                "select_evidence": retry_select_evidence,
                "candidate_fragment_reconciliations": (
                    retry_fragment_reconciliations
                ),
            }
        )
        if not retry_geometry_stable or retry_selection is None:
            raise LiveSelectionError(
                "initial three-frame Live card geometry is unstable",
                events=selection_events,
            )
        initial_selection = retry_selection
        if initial_selection.accepted:
            upgrade_images = retry_images
            identity_source_image = retry_images[-1]
            identity_source_frame_id = frame_identifier(identity_source_image)
            identity_source_candidates = retry_frame_candidates[-1]
            identity_frame_candidates = retry_frame_candidates
            identity_source_event = selection_events[-1]
            identity_source_kind = "retry"
            break
    if not initial_selection.accepted:
        raise LiveSelectionError(
            "initial three-frame Live SELECT state is ambiguous",
            events=selection_events,
        )
    candidates = identity_source_candidates
    image = identity_source_image
    frame_id = identity_source_frame_id
    identity_source_event["identity_source_frame_id"] = frame_id
    identity_source_event["identity_source_candidate_boxes"] = [
        list(candidate.box) for candidate in candidates
    ]
    if (
        expected_initial_hand_count is not None
        and len(candidates) != expected_initial_hand_count
    ):
        identity_source_event.update(
            {
                "identity_source_accepted": False,
                "identity_source_reason": "initial_live_expected_hand_count_mismatch",
                "expected_hand_count": expected_initial_hand_count,
                "observed_hand_count": len(candidates),
            }
        )
        raise LiveSelectionError(
            "initial_live_expected_hand_count_mismatch",
            events=selection_events,
        )
    try:
        predictions = list(
            classify_live_card_views(
                recognizer,
                image,
                candidates,
                frame_id=frame_id,
                plan=plan,
                acceptance_threshold=identity_threshold,
                min_margin=identity_min_margin,
                art_preprocessing_evidence=identity_source_event.setdefault('art_preprocessing_evidence', []),
                **_live_card_art_profile(context, candidates, initial_selection.selected_slot),
            )
        )
    except LiveCardLayoutError as error:
        identity_source_event.update(
            {
                "identity_source_accepted": False,
                "identity_source_reason": "live_identity_source_layout_invalid",
            }
        )
        raise LiveSelectionError(
            f"{identity_source_kind} Live identity source layout is invalid",
            events=selection_events,
        ) from error
    raw_identity_predictions = tuple(predictions)
    predictions = list(_corroborate_bound_hif_families(context,image,candidates,predictions,
        selected_slot=initial_selection.selected_slot,minimum_margin=identity_min_margin,
        evidence=identity_source_event,selected_neighbor_frames=upgrade_images,
        selected_neighbor_candidates=identity_frame_candidates,
        minimum_confidence=params.get('minimum_identity_confidence')))
    predictions = list(_corroborate_recovered_support_family(context, upgrade_images, predictions,
        baseline=selected_resource_baseline, selected_slot=initial_selection.selected_slot,
        minimum_margin=identity_min_margin, evidence=identity_source_event))
    selected_journal = getattr(context, 'gkh_hif_gray_selected_source', None)
    if (initial_selection.selected_slot is not None and isinstance(selected_journal, dict)
            and (not predictions[initial_selection.selected_slot].accepted
                 or predictions[initial_selection.selected_slot].confidence
                    < (params.get('minimum_identity_confidence') or 0))):
        predictions = list(_corroborate_bound_hif_selected_family(context, upgrade_images,
            predictions, selected_journal.get('identity_certificate', ()),
            selected_slot=initial_selection.selected_slot, minimum_margin=identity_min_margin,
            evidence=identity_source_event, catalog=catalog, plan=plan,
            identity_frame_id=frame_id, candidates=candidates,
            minimum_confidence=params.get('minimum_identity_confidence')))
    if (identity_source_kind == 'initial' and initial_selection.selected_slot is None
            and any(not p.accepted for p in predictions)):
        alternative = _existing_unselected_identity_source(context, upgrade_images, initial_frame_candidates,
            raw_identity_predictions, predictions, recognizer=recognizer, plan=plan,
            threshold=identity_threshold, margin=identity_min_margin, evidence=identity_source_event)
        if alternative is not None:
            image, candidates, predictions = alternative
            frame_id = frame_identifier(image)
            identity_source_kind = 'same_batch_whole_hand'
            identity_source_event.update(identity_source_frame_id=frame_id,
                identity_source_candidate_boxes=[list(c.box) for c in candidates],
                identity_source_reselection_reason='inspected_target_transient_first_frame')
        elif input_deadline is None:
            observed = _observe_initial_identity_window(context, upgrade_images, recognizer=recognizer,
                threshold=identity_threshold, margin=identity_min_margin, evidence=identity_source_event,
                observation_started_at=identity_observation_started_at)
            if observed is not None:
                upgrade_images, core = observed
                image, candidates = upgrade_images[-1], core['candidates']
                frame_id = frame_identifier(image)
                identity_source_event.update(phase='initial_identity_rejected', identity_source_accepted=False,
                    identity_source_reason='initial_inspected_target_transient')
                identity_source_event = dict(phase='initial_observation', click_dispatched=False,
                    geometry_stable=True, candidate_boxes=core.get('candidate_boxes', []),
                    frame_ids=list(core['frame_ids']), select_evidence=core['select_evidence'],
                    fields=core['fields'], families=core['families'], identity_source_frame_id=frame_id,
                    identity_source_candidate_boxes=[list(c.box) for c in candidates],
                    identity_source_reselection_reason='complete_new_unselected_group')
                selection_events.append(identity_source_event)
                predictions = classify_live_card_views(recognizer, image, candidates, frame_id=frame_id,
                    plan=plan, acceptance_threshold=identity_threshold, min_margin=identity_min_margin,
                    **_live_card_art_profile(context, candidates, None))
                predictions = list(_corroborate_bound_hif_families(context, image, candidates, predictions,
                    selected_slot=None, minimum_margin=identity_min_margin, evidence=identity_source_event,
                    minimum_confidence=params.get('minimum_identity_confidence')))
                from live_decision.hif_bound_title import family, event_proof

                if any(not p.accepted or family(p, event_proof(identity_source_event, p)) != core['families'][i]
                       for i, p in enumerate(predictions)):
                    raise LiveSelectionError('new complete identity source is unresolved', events=selection_events)
                identity_source_kind = 'complete_new_unselected_group'
    if (
        len(predictions) != len(candidates)
        or tuple(prediction.slot for prediction in predictions)
        != tuple(candidate.slot for candidate in candidates)
    ):
        identity_source_event.update(
            {
                "identity_source_accepted": False,
                "identity_source_reason": "live_identity_source_incomplete",
            }
        )
        raise LiveSelectionError(
            f"{identity_source_kind} Live identity source is incomplete",
            events=selection_events,
        )
    identity_source_event.update(
        {
            "identity_source_accepted": True,
            "identity_source_reason": "live_identity_source_classified",
        }
    )
    current_selected_slot = initial_selection.selected_slot
    initial_selected_slot = current_selected_slot
    identity_provenance = {
        prediction.slot: {
            "source_selected_slot": initial_selected_slot,
            "unselected_view_verified": prediction.slot != initial_selected_slot,
            "selected_transition_verified": False,
            "source_frame_id": prediction.frame_id,
        }
        for prediction in predictions
    }
    for proof in identity_source_event.get('inspected_title_art_families', ()):
        identity_provenance[proof['slot']]['inspected_art_evidence'] = proof
    for proof in identity_source_event.get('bound_complete_title_families', ()):
        identity_provenance[proof['slot']]['bound_title_evidence'] = proof
    if (initial_selected_slot is not None
            and isinstance(getattr(context, 'gkh_hif_gray_selected_source', None), dict)):
        neighbor = initial_selected_slot + 1
        if neighbor < len(predictions) and not predictions[neighbor].accepted:
            try:
                carried, proof = _carry_hif_selected_neighbor(context, predictions[neighbor],
                    evidence=identity_source_event, identity_threshold=identity_threshold,
                    identity_min_margin=identity_min_margin)
            except ValueError as error:
                identity_source_event['selected_neighbor_carry_rejection'] = str(error)
            else:
                predictions[neighbor] = carried
                identity_provenance[neighbor].update(source_selected_slot=None,
                    unselected_view_verified=True, source_frame_id=carried.frame_id)
                if proof.get('source_title_evidence') is not None:
                    identity_provenance[neighbor].update(source_selected_slot=initial_selected_slot,
                        unselected_view_verified=False, selected_neighbor_identity_carry=proof)
                selection_events.append(dict(phase='selected_neighbor_identity_carry', **proof))
    title_views = {
        view.slot: view
        for view in build_live_title_views(
            tuple(image.shape),
            candidates,
            selected_slot=current_selected_slot,
            hif_layout=isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext),
        )
    }
    deferred_selected_upgrade: LiveUpgradeConsensus | None = None
    if (
        live_mode in {"selection_observe", "debug_execute"}
        and initial_selected_slot is not None
        and not predictions[initial_selected_slot].accepted
    ):
        try:
            deferred_selected_upgrade = resolve_live_title_consensus(
                image,
                title_views[initial_selected_slot],
                frame_id=frame_id,
            )
        except ValueError:
            deferred_selected_upgrade = LiveUpgradeConsensus(
                accepted=False,
                upgraded=None,
                plus_votes=0,
                normal_votes=0,
                reason="live_upgrade_slot_not_observable",
                observations=(),
            )
    upgrades: dict[int, LiveUpgradeConsensus] = {}
    requires_upgrade: dict[int, bool] = {}
    family_ids_by_slot: dict[int, tuple[str, ...]] = {}
    from live_decision.hif_bound_title import family as identity_family

    for candidate, prediction in zip(candidates, predictions, strict=True):
        family_ids = identity_family(prediction,
            identity_provenance[candidate.slot].get('bound_title_evidence')
            or identity_provenance[candidate.slot].get('inspected_art_evidence'))
        family_ids_by_slot[candidate.slot] = family_ids
        eligible_ids = _eligible_family_ids(catalog, family_ids, plan)
        requires_upgrade[candidate.slot] = len(eligible_ids) > 1
        if len(eligible_ids) <= 1:
            upgrades[candidate.slot] = LiveUpgradeConsensus(
                accepted=True,
                upgraded=None,
                plus_votes=0,
                normal_votes=0,
                reason="live_upgrade_not_required",
                observations=(),
            )
            continue
        try:
            upgrades[candidate.slot] = resolve_live_title_consensus(
                image,
                title_views[candidate.slot],
                frame_id=frame_id,
            )
        except ValueError:
            upgrades[candidate.slot] = LiveUpgradeConsensus(
                accepted=False,
                upgraded=None,
                plus_votes=0,
                normal_votes=0,
                reason="live_upgrade_slot_not_observable",
                observations=(),
            )

    selection_click_count = 0
    if (initial_selected_slot is not None
            and (len(candidates) in (4, 5) or (len(candidates) in (1, 2, 3)
                 and getattr(context, 'gkh_hif_opening_state', None) is not None))
            and isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext)
            and predictions[initial_selected_slot].accepted):
        selected_proof = identity_provenance[initial_selected_slot].get('bound_title_evidence', {})
        if selected_proof.get('reason') == 'bound_fresh_selected_support_receipt':
            # Both receipt and detail were already read from these exact three
            # frames by the resource baseline; do not OCR the title twice.
            detail_identity = selected_resource_baseline['detail_identity']
            visible_id = selected_proof['selected_support_identity']['visible_card_id']
            if detail_identity.get('accepted') is not True or detail_identity.get('card_id') != visible_id:
                raise LiveSelectionError('fresh selected support detail identity changed')
            upgraded = catalog.entries[visible_id].upgraded
            selected_upgrade = LiveUpgradeConsensus(accepted=True, upgraded=upgraded,
                plus_votes=3 if upgraded else 0, normal_votes=0 if upgraded else 3,
                reason='hif_complete_title_unanimous', observations=())
        else:
            selected_upgrade, detail_identity = _hif_selected_upgrade_consensus(
                context, upgrade_images, family_ids_by_slot[initial_selected_slot], catalog, plan)
        selection_events.append({"phase": "hif_selected_detail_identity",
                                 "slot": initial_selected_slot, **detail_identity})
        if not selected_upgrade.accepted:
            predictions[initial_selected_slot] = replace(
                predictions[initial_selected_slot], accepted=False,
                card_id=UNKNOWN, reason="hif_complete_title_unresolved")
        upgrades[initial_selected_slot] = selected_upgrade
    if (initial_selected_slot is not None and len(candidates) in (1, 2, 3)
            and isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext)
            and getattr(context, 'gkh_hif_opening_state', None) is not None
            and all(p.accepted for p in predictions)):
        from live_decision.opening_observer import (
            read_opening_identity_titles,
            selected_short_hand_title_card_ids,
            verify_recovered_short_hand_titles,
            retained_selected_short_hand_titles,
        )

        title_frames = tuple(read_opening_identity_titles(context, frame) for frame in upgrade_images)
        retained_titles = None
        try:
            full_ids = selected_short_hand_title_card_ids(
                title_frames,
                tuple(family_ids_by_slot[c.slot] for c in candidates), catalog,
                selected_slot=initial_selected_slot, selected_detail=detail_identity,
                frame_ids=tuple(frame_identifier(f) for f in upgrade_images))
            if (len(candidates) != 3
                    and getattr(context, 'gkh_hif_gray_selected_source', None) is not None):
                verify_recovered_short_hand_titles(context.gkh_hif_opening_state, full_ids)
        except ValueError as error:
            recovery = getattr(context, 'gkh_hif_gray_selected_source', None)
            if (str(error) != 'HIF opening hand title confidence is unresolved'
                    or len(candidates) not in (2,3) or not isinstance(recovery,dict)
                    or recovery.get('ordinary_safe_preview') is not True
                    or recovery.get('selection_ready') is not True or recovery.get('business_submit_count') != 0
                    or recovery.get('refresh_a',{}).get('proposal',{}).get('candidate_traces',{}).get(
                        'high_score',{}).get('action',{}).get('source_slot') != initial_selected_slot):
                raise LiveSelectionError('hif_selected_full_title_identity_unresolved:'+str(error),
                                         events=selection_events) from error
            try:
                families = tuple(family_ids_by_slot[c.slot] for c in candidates)
                full_ids = _retained_hif_hand_ids(context,families)
                retained_titles = retained_selected_short_hand_titles(title_frames,families,full_ids,catalog,upgrades,
                    selected_slot=initial_selected_slot,selected_detail=detail_identity,
                    frame_ids=tuple(frame_identifier(f) for f in upgrade_images))
                verify_recovered_short_hand_titles(context.gkh_hif_opening_state,full_ids)
            except ValueError as retained_error:
                raise LiveSelectionError('hif_selected_retained_title_identity_unresolved:'+str(retained_error),
                                         events=selection_events) from retained_error
        for candidate,card_id in zip(candidates,full_ids,strict=True):
            if retained_titles is not None:
                continue
            upgraded = catalog.entries[card_id].upgraded
            upgrades[candidate.slot] = LiveUpgradeConsensus(accepted=True, upgraded=upgraded,
                plus_votes=3 if upgraded else 0, normal_votes=0 if upgraded else 3,
                reason='hif_three_frame_complete_title',observations=())
        selection_events.append(dict(phase='hif_selected_complete_title_identity' if retained_titles is None
            else 'hif_selected_retained_title_identity', accepted=True,card_ids=full_ids,
            frame_ids=[frame_identifier(f) for f in upgrade_images],
            **({'retained_title_evidence':retained_titles} if retained_titles is not None else {})))
    if (initial_selected_slot is None and len(candidates) in (1, 2, 3)
            and getattr(context, 'gkh_hif_opening_state', None) is not None):
        from live_decision.opening_observer import (
            opening_title_card_ids,
            retained_short_hand_titles,
            read_opening_identity_titles,
        )

        retained_titles = None
        try:
            title_frames = tuple(read_opening_identity_titles(context, frame)
                                 for frame in upgrade_images)
            full_ids = opening_title_card_ids(title_frames,
                tuple(family_ids_by_slot[c.slot] for c in candidates), catalog)
        except ValueError as error:
            if (str(error) != 'HIF opening hand title confidence is unresolved'
                    or len(candidates) not in (1,2,3) or not all(p.accepted for p in predictions)):
                raise LiveSelectionError('hif_full_title_identity_unresolved:'+str(error),
                                         events=selection_events) from error
            try:
                full_ids = _retained_hif_hand_ids(context,
                    tuple(family_ids_by_slot[c.slot] for c in candidates))
                retained_titles = retained_short_hand_titles(title_frames,full_ids,catalog,upgrades)
            except ValueError as retained_error:
                raise LiveSelectionError('hif_retained_title_identity_unresolved:'+str(retained_error),
                                         events=selection_events) from retained_error
        for candidate,card_id in zip(candidates,full_ids,strict=True):
            if retained_titles is not None:
                continue
            upgraded = catalog.entries[card_id].upgraded
            upgrades[candidate.slot] = LiveUpgradeConsensus(accepted=True, upgraded=upgraded,
                plus_votes=3 if upgraded else 0, normal_votes=0 if upgraded else 3,
                reason='hif_three_frame_complete_title',observations=())
        selection_events.append(dict(phase='hif_complete_title_identity' if retained_titles is None
            else 'retained_complete_hand_identity',accepted=True,card_ids=full_ids,
            frame_ids=[frame_identifier(f) for f in upgrade_images],
            **({'retained_title_evidence':retained_titles} if retained_titles is not None else {})))
        if retained_titles is not None and len(candidates) == 2:
            selection_events.append(dict(phase='retained_complete_hand_identity', accepted=True,
                card_ids=full_ids, run_id=context.gkh_hif_live_context.run_id,
                revision=context.gkh_hif_live_context.revision))
    selected_recovery = getattr(context, 'gkh_hif_gray_selected_source', None)
    retained_selected_hand = (isinstance(selected_recovery, dict)
        and selected_recovery.get('ordinary_safe_preview') is True
        and selected_recovery.get('selection_ready') is True
        and selected_recovery.get('business_submit_count', 0) == 0
        and selected_recovery.get('refresh_a', {}).get('proposal', {}).get(
            'candidate_traces', {}).get('high_score', {}).get('action', {}).get(
                'source_slot') == initial_selected_slot)
    if (initial_selected_slot is not None and len(candidates) in (4, 5)
            and isinstance(selected_recovery, dict)
            and selected_recovery.get('ordinary_safe_preview') is not True):
        from live_decision.hif_gray_selection import selected_zero_cost_baseline

        if (not isinstance(selected_resource_baseline, dict)
                or selected_resource_baseline.get('accepted') is not True
                or selected_resource_baseline.get('selected_slot') != initial_selected_slot
                or selected_resource_baseline.get('frame_ids') != [frame_identifier(f) for f in upgrade_images]):
            raise LiveSelectionError('retained gray hand lacks this observation baseline')
        try:
            selected_zero_cost_baseline(context.gkh_hif_opening_state,
                complete_event=context.gkh_hif_committed_event, journal=selected_recovery,
                evidence=selected_resource_baseline, geometry_comparator=_same_live_candidate_geometry)
        except ValueError as error:
            raise LiveSelectionError('retained gray hand: '+str(error)) from error
        retained_selected_hand = True
    temporary_hand_support = {}
    if ((initial_selected_slot is None or retained_selected_hand) and len(candidates) in (1, 2, 3, 4, 5)
            and getattr(context, 'gkh_hif_opening_state', None) is not None
            # The fresh three-card opening was just resolved from complete
            # titles above; it has no committed settlement to retain yet.
            # Invalid settlements and larger/selected hands still fail closed.
            and not (initial_selected_slot is None and len(candidates) == 3
                     and context.gkh_hif_opening_state.state.turn == 1
                     and getattr(context, 'gkh_hif_committed_state', None) is None
                     and getattr(context, 'gkh_hif_committed_event', None) is None)
            and all(p.accepted for p in predictions)):
        ids = _retained_hif_hand_ids(context, tuple(family_ids_by_slot[c.slot] for c in candidates))
        for candidate, card_id in zip(candidates, ids, strict=True):
            old_upgrade = upgrades[candidate.slot]
            upgraded = catalog.entries[card_id].upgraded
            if old_upgrade.accepted and old_upgrade.upgraded is not None and old_upgrade.upgraded is not upgraded:
                if (not upgraded and old_upgrade.upgraded is True
                        and retained_selected_hand and candidate.slot == initial_selected_slot):
                    from live_decision.hif_gray_selection import recovered_selected_support_hand_identity

                    try:
                        proof = recovered_selected_support_hand_identity(
                            context.gkh_hif_opening_state,
                            complete_event=context.gkh_hif_committed_event,
                            journal=selected_recovery, baseline=selected_resource_baseline,
                            frame_ids=tuple(frame_identifier(f) for f in upgrade_images),
                            slot=candidate.slot)
                    except ValueError as error:
                        raise LiveSelectionError('visible HIF upgrade contradicts complete hand: '+str(error)) from error
                elif (upgraded or old_upgrade.upgraded is not True
                        or initial_selected_slot is not None
                        and (candidate.slot == initial_selected_slot
                             or len(candidates) == 5 and candidate.slot == initial_selected_slot+1)):
                    raise LiveSelectionError('visible HIF upgrade contradicts complete hand')
                else:
                    from live_decision.opening_support import observe_bound_rightmost_support

                    try:
                        proof = observe_bound_rightmost_support(context, upgrade_images,
                            bundle=context.gkh_hif_opening_state, slot=candidate.slot,
                            frame_identifier=frame_identifier, candidates=candidates,
                            selected_slot=initial_selected_slot)
                    except ValueError as error:
                        raise LiveSelectionError('visible HIF upgrade contradicts complete hand: '+str(error)) from error
                temporary_hand_support[candidate.slot] = proof
                identity_provenance[candidate.slot]['temporary_support_identity'] = proof
                # A selected-action projection is bound by its own receipt;
                # it must not enter the initial whole-hand projection path.
                phase = ('hif_selected_ordinary_plus_identity'
                         if proof.get('source') == 'fresh_selected_ordinary_plus_baseline'
                         else 'hif_temporary_support_identity')
                selection_events.append(dict(phase=phase, **proof))
            if not old_upgrade.accepted and old_upgrade.reason == 'live_upgrade_suffix_occluded':
                upgrades[candidate.slot] = LiveUpgradeConsensus(accepted=True, upgraded=upgraded,
                    plus_votes=0, normal_votes=0, reason='retained_complete_hand_occluded_suffix', observations=())
        selection_events.append(dict(phase='retained_complete_hand_identity', accepted=True, card_ids=ids,
                                     run_id=context.gkh_hif_live_context.run_id,
                                     revision=context.gkh_hif_live_context.revision))
    if (initial_selected_slot is not None and len(candidates) == 3
            and isinstance(getattr(context, 'gkh_hif_live_context', None), HifLiveRunContext)
            and getattr(context, 'gkh_hif_opening_state', None) is not None
            and isinstance(getattr(context, 'gkh_hif_gray_selected_source', None), dict)
            and all(p.accepted for p in predictions)):
        # A blue suffix names the temporary visible form. Only the current
        # three-frame support proof can map it back to its permanent entity.
        from live_decision.opening_observer import verify_recovered_short_hand_titles

        normalized_ids = tuple(temporary_hand_support.get(candidate.slot, {}).get('card_id', card_id)
                               for candidate, card_id in zip(candidates, full_ids, strict=True))
        try:
            verify_recovered_short_hand_titles(context.gkh_hif_opening_state, normalized_ids)
        except ValueError as error:
            raise LiveSelectionError('hif_selected_full_title_identity_unresolved:'+str(error),
                                     events=selection_events) from error
    unresolved = [
        candidate
        for candidate, prediction in zip(candidates, predictions, strict=True)
        if prediction.accepted
        and requires_upgrade[candidate.slot]
        and not upgrades[candidate.slot].accepted
    ]
    if (unresolved and live_mode == "selection_observe" and current_selected_slot is None
            and isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext)
            and all(prediction.accepted for prediction in predictions)):
        from live_decision.inventory import match_inventory_hand

        try:
            inventory_ids, returned_frames, returned, inventory_events = _observe_hif_inventory_transaction(
                context, recognizer, maximum_age_ms=params["max_snapshot_age_ms"],
                identity_threshold=identity_threshold, identity_min_margin=identity_min_margin)
        except LiveSelectionError as error:
            raise LiveSelectionError(str(error), events=[*selection_events, *error.events]) from error
        selection_events.extend(inventory_events)
        selection_click_count += sum(event.get("input_click_count", 0) for event in inventory_events)
        if (returned["families"] != tuple(family_ids_by_slot[candidate.slot] for candidate in candidates)
                or not _same_live_candidate_geometry(candidates, returned["candidates"])):
            raise LiveSelectionError("inventory return does not match the identity source", events=selection_events)
        constrained_families = []
        for candidate in candidates:
            family = family_ids_by_slot[candidate.slot]
            upgrade = upgrades[candidate.slot]
            known = (catalog.resolve_upgrade_state(family, upgraded=upgrade.upgraded, plan=plan)
                     if upgrade.accepted and upgrade.upgraded is not None else None)
            constrained_families.append((known,) if known else _eligible_family_ids(catalog, family, plan))
        try:
            mapped_ids = match_inventory_hand(constrained_families, inventory_ids)
        except ValueError as error:
            raise LiveSelectionError(str(error), events=selection_events) from error
        candidates = returned["candidates"]
        upgrade_images = returned_frames
        image, frame_id = returned_frames[-1], frame_identifier(returned_frames[-1])
        for candidate, card_id in zip(candidates, mapped_ids, strict=True):
            upgraded = catalog.entries[card_id].upgraded
            upgrades[candidate.slot] = LiveUpgradeConsensus(
                accepted=True, upgraded=upgraded, plus_votes=3 if upgraded else 0,
                normal_votes=0 if upgraded else 3, reason="inventory_three_frame_unique_hand_assignment",
                observations=())
            predictions[candidate.slot] = replace(predictions[candidate.slot], frame_id=frame_id,
                source_frames=returned["frame_ids"], box=candidate.box)
            identity_provenance[candidate.slot].update({
                "source_frame_id": frame_id, "inventory_return_verified": True,
                "inventory_card_id": card_id})
        selection_events.append({"phase": "inventory_hand_assignment", "accepted": True,
                                 "mapped_ids": mapped_ids, "return_frame_ids": returned["frame_ids"]})
        unresolved = []
    selectable_unresolved = (
        unresolved
        if len(candidates) == 4
        and live_mode == "selection_observe"
        else ()
    )
    identity_refresh_frame = None
    identity_refresh_candidates: tuple[Any, ...] | None = None
    for unresolved_candidate in selectable_unresolved:
        candidate = candidates[unresolved_candidate.slot]
        current_input_deadline = input_deadline
        current_authorization_floor: float | None = None
        if live_mode in {"selection_observe", "debug_execute"}:
            candidates, current_input_deadline, preflight_event = (
                _selection_observation_preflight(
                    context,
                    candidates,
                    current_selected_slot,
                    candidate.slot,
                    float(params["max_snapshot_age_ms"]),
                )
            )
            selection_events.append(preflight_event)
            if not preflight_event["accepted"]:
                raise LiveSelectionError(
                    "Live selection-observation preflight failed",
                    events=selection_events,
                )
            preflight_floor = preflight_event.get("authorization_floor")
            if type(preflight_floor) in {int, float}:
                current_authorization_floor = float(preflight_floor)
            candidate = candidates[unresolved_candidate.slot]
        deadline_expired, observed_floor = _validate_live_input_clock_window(
            current_input_deadline,
            current_authorization_floor,
        )
        if deadline_expired:
            selection_events.append(
                {
                    "phase": "selection_observation",
                    "attempted_slot": candidate.slot,
                    "click_dispatched": False,
                    "frame_ids": [],
                    "select_consensus": None,
                    "reason": "snapshot_expired_before_input",
                }
            )
            raise LiveSelectionError(
                "Live snapshot expired before selection observation input",
                events=selection_events,
            )
        current_authorization_floor = observed_floor
        if current_selected_slot == candidate.slot:
            raise LiveSelectionError(
                "refusing to click the currently selected Live card",
                events=selection_events,
            )
        deadline_expired, observed_floor = _validate_live_input_clock_window(
            current_input_deadline,
            current_authorization_floor,
        )
        if deadline_expired:
            selection_events.append(
                {
                    "phase": "selection_observation",
                    "attempted_slot": candidate.slot,
                    "click_dispatched": False,
                    "frame_ids": [],
                    "select_consensus": None,
                    "reason": "snapshot_expired_before_input",
                }
            )
            raise LiveSelectionError(
                "Live snapshot expired before selection observation input",
                events=selection_events,
            )
        if not _click_live_candidate(context, candidate):
            selection_events.append(
                {
                    "phase": "selection_observation",
                    "attempted_slot": candidate.slot,
                    "click_dispatched": False,
                    "frame_ids": [],
                    "select_consensus": None,
                }
            )
            raise LiveSelectionError(
                "Maa Live selection click failed",
                events=selection_events,
            )
        selection_click_count += 1
        time.sleep(SELECTION_SETTLE_SECONDS)
        confirmation = _capture_selected_live_confirmation(
            context,
            expected_hand_size=len(candidates),
            target_slot=candidate.slot,
        )
        selection_events.extend(confirmation["events"])
        if not confirmation["accepted"]:
            raise LiveSelectionError(
                "three-frame Live geometry/SELECT evidence did not match the clicked slot",
                events=selection_events,
            )
        identity_provenance[candidate.slot]["selected_transition_verified"] = True
        identity_provenance[candidate.slot]["selection_transition_frame_ids"] = [
            frame_identifier(frame) for frame in confirmation["frames"]
        ]
        selected_frames = confirmation["frames"]
        selected_frame_candidates = confirmation["frame_candidates"]
        identity_refresh_frame = selected_frames[-1]
        identity_refresh_candidates = selected_frame_candidates[-1]
        selection_event = confirmation["events"][-1]
        candidates = selected_frame_candidates[-1]
        current_selected_slot = candidate.slot
        selected_view = build_live_title_views(
            tuple(selected_frames[-1].shape),
            candidates,
            selected_slot=current_selected_slot,
        )[current_selected_slot]
        selected_title_frame = selected_frames[-1]
        upgrade = resolve_live_title_consensus(
            selected_title_frame,
            selected_view,
            frame_id=frame_identifier(selected_title_frame),
        )
        if isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext):
            upgrade, detail = _hif_selected_upgrade_consensus(
                context, selected_frames, family_ids_by_slot[candidate.slot], catalog, plan)
            selection_event["hif_selected_detail_identity"] = detail
        selection_event["upgrade_consensus"] = upgrade.to_dict()
        if not upgrade.accepted:
            raise LiveSelectionError(
                "selected Live title did not resolve enhancement",
                events=selection_events,
            )
        upgrades[candidate.slot] = upgrade

    # A card that was already selected when the first frame was captured can
    # be enlarged enough to invalidate the overlapping-hand art crop.  Once a
    # verified upgrade probe has selected another slot, reclassify only that
    # formerly selected card from the now-unselected geometry.  This does not
    # dispatch another click and keeps the original single-frame suffix evidence
    # for resolving its normal/+ member after the visual family is recovered.
    if (
        live_mode in {"selection_observe", "debug_execute"}
        and initial_selected_slot is not None
        and not predictions[initial_selected_slot].accepted
        and current_selected_slot != initial_selected_slot
        and identity_refresh_frame is not None
        and identity_refresh_candidates is not None
        and len(identity_refresh_candidates) == 4
    ):
        try:
            refreshed_predictions = classify_live_card_views(
                recognizer,
                identity_refresh_frame,
                identity_refresh_candidates,
                frame_id=frame_identifier(identity_refresh_frame),
                plan=plan,
                acceptance_threshold=identity_threshold,
                min_margin=identity_min_margin,
                **_live_card_art_profile(context, identity_refresh_candidates, current_selected_slot),
            )
        except LiveCardLayoutError:
            refreshed_predictions = ()
        refreshed_by_slot = {
            prediction.slot: prediction for prediction in refreshed_predictions
        }
        refreshed_prediction = refreshed_by_slot.get(initial_selected_slot)
        refresh_event: dict[str, Any] = {
            "phase": "initial_selected_identity_refresh",
            "attempted_slot": initial_selected_slot,
            "click_dispatched": False,
            "frame_ids": [frame_identifier(identity_refresh_frame)],
            "selected_slot_during_refresh": current_selected_slot,
            "prediction_accepted": bool(
                refreshed_prediction is not None and refreshed_prediction.accepted
            ),
            "accepted": False,
            "prediction": (
                refreshed_prediction.to_dict()
                if refreshed_prediction is not None
                else None
            ),
        }
        if refreshed_prediction is not None and refreshed_prediction.accepted:
            predictions[initial_selected_slot] = refreshed_prediction
            identity_provenance[initial_selected_slot] = {
                "source_selected_slot": current_selected_slot,
                "unselected_view_verified": (
                    current_selected_slot is not None
                    and current_selected_slot != initial_selected_slot
                ),
                "selected_transition_verified": False,
                "source_frame_id": refreshed_prediction.frame_id,
            }
            family_ids = (
                refreshed_prediction.top_k_card_ids[0]
                if refreshed_prediction.top_k_card_ids
                else ()
            )
            family_ids_by_slot[initial_selected_slot] = family_ids
            eligible_ids = _eligible_family_ids(catalog, family_ids, plan)
            requires_upgrade[initial_selected_slot] = len(eligible_ids) > 1
            if len(eligible_ids) <= 1:
                upgrades[initial_selected_slot] = LiveUpgradeConsensus(
                    accepted=True,
                    upgraded=None,
                    plus_votes=0,
                    normal_votes=0,
                    reason="live_upgrade_not_required",
                    observations=(),
                )
                refresh_event.update(
                    {
                        "accepted": True,
                        "reason": "initial_selected_identity_reclassified",
                    }
                )
            else:
                refreshed_upgrade = deferred_selected_upgrade
                if refreshed_upgrade is None:
                    refreshed_upgrade = LiveUpgradeConsensus(
                        accepted=False,
                        upgraded=None,
                        plus_votes=0,
                        normal_votes=0,
                        reason="live_upgrade_slot_not_observable",
                        observations=(),
                    )
                upgrades[initial_selected_slot] = refreshed_upgrade
                refresh_event["upgrade_consensus"] = refreshed_upgrade.to_dict()
                refresh_event.update(
                    {
                        "accepted": refreshed_upgrade.accepted,
                        "reason": (
                            "initial_selected_identity_reclassified"
                            if refreshed_upgrade.accepted
                            else "initial_selected_upgrade_unresolved"
                        ),
                    }
                )
        else:
            refresh_event["reason"] = (
                "initial_selected_identity_rejected"
                if refreshed_prediction is not None
                else "initial_selected_identity_layout_invalid"
            )
        selection_events.append(refresh_event)

    observed: list[dict[str, Any]] = []
    for candidate, prediction in zip(candidates, predictions, strict=True):
        upgrade = upgrades[candidate.slot]
        family_ids = family_ids_by_slot[candidate.slot]
        eligible_ids = _eligible_family_ids(catalog, family_ids, plan)
        if prediction.accepted and len(eligible_ids) == 1:
            resolved_id = eligible_ids[0]
        elif prediction.accepted and candidate.slot in temporary_hand_support:
            resolved_id = temporary_hand_support[candidate.slot]['card_id']
        elif prediction.accepted and upgrade.accepted and upgrade.upgraded is not None:
            resolved_id = catalog.resolve_upgrade_state(
                family_ids,
                upgraded=upgrade.upgraded,
                plan=plan,
            )
        else:
            resolved_id = None
        entry = catalog.entries.get(str(resolved_id)) if resolved_id is not None else None
        resolved_cost = entry.stamina_cost if entry is not None else None
        identity_accepted = entry is not None and resolved_cost is not None
        enhancement_tag = (
            "plus"
            if entry is not None and entry.upgraded
            else "normal"
            if entry is not None
            else "unknown"
        )
        prediction = replace(
            prediction,
            box=candidate.box,
            card_id=str(resolved_id) if identity_accepted else UNKNOWN,
            predicted_card_id=(
                str(resolved_id) if resolved_id is not None else prediction.predicted_card_id
            ),
            class_name=(
                f"{entry.card_id}_{entry.name}" if entry is not None else prediction.class_name
            ),
            accepted=identity_accepted,
            reason=(
                "live_binary_upgrade_resolved"
                if identity_accepted and requires_upgrade[candidate.slot]
                else "live_upgrade_not_required"
                if identity_accepted
                else "live_upgrade_state_unresolved"
                if not upgrade.accepted
                else prediction.reason
            ),
        )
        observed.append(
            {
                "prediction": prediction,
                "cost": resolved_cost,
                "playable": candidate.detector_label in {"cards", "suggestions"},
                "features": (
                    catalog.features(str(resolved_id), turns_remaining=turns_remaining)
                    if identity_accepted
                    else None
                ),
                "tags": [
                    f"plan:{plan}",
                    f"card_type:{entry.card_type}" if entry is not None else "card_type:unknown",
                    "identity:visual-family",
                    f"enhancement:{enhancement_tag}",
                    *(["transaction:unsupported_typed_cost"]
                      if entry is not None and not entry.stamina_transaction_supported else []),
                ],
                "upgraded": entry.upgraded if entry is not None else None,
                "upgrade_evidence": upgrade.to_dict(),
                "identity_provenance": identity_provenance[candidate.slot],
            }
        )
    return observed, current_selected_slot, selection_click_count, selection_events


def _execute_selected_live_card(
    context: Context,
    selected: Any,
    hand: tuple[Any, ...],
    current_selected_slot: int | None,
    before_state: Any | None = None,
    input_deadline: float | None = None,
) -> tuple[bool, int, list[dict[str, Any]]]:
    """Send only the remaining clicks needed from the verified SELECT state."""

    required_clicks = remaining_live_card_clicks(
        current_selected_slot,
        selected.slot,
        hand_size=len(hand),
    )
    click_count = 0
    preflight_frames = _capture_three_frames(context)
    preflight_consensus = resolve_live_selected_slot_consensus(
        _locate_selected_live_slots(context, frame, hand) for frame in preflight_frames
    )
    selection_events: list[dict[str, Any]] = [
        {
            "phase": "execution_preflight",
            "attempted_slot": None,
            "click_dispatched": False,
            "frame_ids": [frame_identifier(frame) for frame in preflight_frames],
            "select_consensus": preflight_consensus.to_dict(),
        }
    ]
    if (
        not preflight_consensus.accepted
        or preflight_consensus.selected_slot != current_selected_slot
    ):
        return False, click_count, selection_events
    if current_selected_slot != selected.slot:
        if _input_deadline_expired(input_deadline):
            selection_events.append(
                {
                    "phase": "execution_selection",
                    "attempted_slot": selected.slot,
                    "click_dispatched": False,
                    "frame_ids": [],
                    "select_consensus": None,
                    "reason": "snapshot_expired_before_input",
                }
            )
            return False, click_count, selection_events
        if not _click_live_candidate(context, selected):
            selection_events.append(
                {
                    "phase": "execution_selection",
                    "attempted_slot": selected.slot,
                    "click_dispatched": False,
                    "frame_ids": [],
                    "select_consensus": None,
                }
            )
            return False, click_count, selection_events
        click_count += 1
        time.sleep(SELECTION_SETTLE_SECONDS)
        frames = _capture_three_frames(context)
        selection_consensus = resolve_live_selected_slot_consensus(
            _locate_selected_live_slots(context, frame, hand) for frame in frames
        )
        selection_events.append(
            {
                "phase": "execution_selection",
                "attempted_slot": selected.slot,
                "click_dispatched": True,
                "frame_ids": [frame_identifier(frame) for frame in frames],
                "select_consensus": selection_consensus.to_dict(),
            }
        )
        if (
            not selection_consensus.accepted
            or selection_consensus.selected_slot != selected.slot
        ):
            return False, click_count, selection_events
        current_selected_slot = selected.slot
    if current_selected_slot != selected.slot:
        return False, click_count, selection_events
    if _input_deadline_expired(input_deadline):
        selection_events.append(
            {
                "phase": "execution_submit",
                "attempted_slot": selected.slot,
                "click_dispatched": False,
                "frame_ids": [],
                "select_consensus": None,
                "reason": "snapshot_expired_before_input",
            }
        )
        return False, click_count, selection_events
    executed = _click_live_candidate(context, selected)
    click_count += 1
    selection_events.append(
        {
            "phase": "execution_submit",
            "attempted_slot": selected.slot,
            "click_dispatched": executed,
            "frame_ids": [],
            "select_consensus": None,
        }
    )
    if not executed or click_count != required_clicks:
        return False, click_count, selection_events
    try:
        committed, postcondition_events = _confirm_live_selection_transition(
            context,
            selected,
            before_state,
        )
    except Exception as error:
        selection_events.append(
            {
                "phase": "execution_postcondition",
                "attempted_slot": selected.slot,
                "click_dispatched": False,
                "frame_ids": [],
                "select_consensus": None,
                "accepted": False,
                "reason": "postcondition_exception",
                "error_type": type(error).__name__,
                "input_may_have_been_dispatched": True,
            }
        )
        return False, click_count, selection_events
    selection_events.extend(postcondition_events)
    return committed, click_count, selection_events


def _recognition_hit(context: Context, node_name: str, image: Any) -> bool:
    if (node_name == 'ProduceButton'
            and isinstance(getattr(context, 'gkh_hif_live_context', None), HifLiveRunContext)):
        from live_decision.popup_guard import parallel_hif_popup_hit
        return parallel_hif_popup_hit(context, image)
    detail = context.run_recognition(node_name, image)
    return bool(detail and detail.hit)


def _observe_post_submit_business_fields(context: Context, image: Any) -> dict[str, Any]:
    """Read positive post-submit business evidence without dispatching input."""

    turns_remaining = _ocr_field(
        context,
        image,
        "turns_remaining",
        LIVE_LAYOUT_V1["turns_remaining"][0],
        LIVE_LAYOUT_V1["turns_remaining"][1],
    )
    stamina = _ocr_field(
        context,
        image,
        "stamina",
        LIVE_LAYOUT_V1["stamina"][0],
        LIVE_LAYOUT_V1["stamina"][1],
    )
    action_order = _locate_live_self_rank(context, image)
    score = (
        _ocr_live_score(
            context,
            image,
            action_order,
        )
        if action_order.get("accepted") is True
        else {"accepted": False, "value": None, "confidence": 0.0}
    )
    return {
        "turns_remaining": turns_remaining,
        "stamina": stamina,
        "action_order": action_order,
        "score": score,
        "no_cards": _recognition_hit(context, "ProduceRecognitionNoCards", image),
        "terminal": {
            node_name: _recognition_hit(context, node_name, image)
            for node_name in LIVE_POST_TERMINAL_NODES
        },
    }


def _observe_post_submit_live_selection(
    context: Context,
    image: Any,
) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[int, ...]] | None:
    """Re-detect the current hand before inspecting SELECT labels."""

    detail = context.run_recognition("ProduceRecognitionCards", image)
    results = list(getattr(detail, "all_results", ()) or ()) if detail and detail.hit else []
    candidates = _normalise_observed_live_candidates(context, image, results)
    if not candidates:
        return None
    geometry = tuple(tuple(candidate.box) for candidate in candidates)
    selected_slots = _locate_selected_live_slots(context, image, candidates)
    return geometry, selected_slots


def _confirm_live_selection_transition(
    context: Context,
    selected: Any,
    before_state: Any | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Observe SELECT state after submit without remedial input.

    SELECT disappearance is telemetry only and cannot return success until a
    positive hand/score/stamina/turn or terminal business delta exists.
    """

    pre_business_values = {
        name: (
            value
            if type(value := getattr(before_state, name, None)) is int
            else None
        )
        for name in ("score", "stamina", "turns_remaining")
    }
    before_hand = getattr(before_state, "hand", ())
    pre_hand_count = len(before_hand) if isinstance(before_hand, (list, tuple)) else None
    events: list[dict[str, Any]] = []
    scene_batch = None
    if isinstance(getattr(context, 'gkh_hif_live_context', None), HifLiveRunContext):
        from live_decision.hif_authoritative_scene import prepare_scene_batch

        scene_batch = prepare_scene_batch(context, _recognition_hit)
    for attempt in range(1, EXECUTION_POST_CONFIRM_ATTEMPTS + 1):
        time.sleep(EXECUTION_POST_SETTLE_SECONDS)
        frames = _capture_three_frames(context)
        business_observations = (
            tuple(_observe_post_submit_business_fields(context, frame) for frame in frames)
            if before_state is not None
            else tuple(
                {
                    "turns_remaining": {
                        "accepted": False,
                        "value": None,
                        "confidence": 0.0,
                    },
                    "stamina": {"accepted": False, "value": None, "confidence": 0.0},
                    "action_order": {
                        "accepted": False,
                        "value": None,
                        "confidence": 0.0,
                    },
                    "score": {"accepted": False, "value": None, "confidence": 0.0},
                    "no_cards": False,
                    "terminal": {
                        node_name: False for node_name in LIVE_POST_TERMINAL_NODES
                    },
                }
                for _frame in frames
            )
        )
        stable_post_values: dict[str, int | None] = {}
        for field_name in ("score", "stamina", "turns_remaining"):
            values = tuple(
                observation[field_name].get("value")
                for observation in business_observations
                if observation[field_name].get("accepted") is True
                and type(observation[field_name].get("value")) is int
            )
            stable_post_values[field_name] = (
                values[0]
                if len(values) == len(frames) and all(value == values[0] for value in values)
                else None
            )
        no_cards_votes = sum(
            observation["no_cards"] is True for observation in business_observations
        )
        terminal_votes = {
            node_name: sum(
                observation["terminal"][node_name] is True
                for observation in business_observations
            )
            for node_name in LIVE_POST_TERMINAL_NODES
        }
        business_deltas: dict[str, Any] = {}
        stable_score = stable_post_values["score"]
        if (
            pre_business_values["score"] is not None
            and stable_score is not None
            and stable_score > pre_business_values["score"]
        ):
            business_deltas["score"] = {
                "before": pre_business_values["score"],
                "after": stable_score,
                "delta": stable_score - pre_business_values["score"],
            }
        stable_turns = stable_post_values["turns_remaining"]
        if (
            pre_business_values["turns_remaining"] is not None
            and stable_turns == pre_business_values["turns_remaining"] - 1
        ):
            business_deltas["turns_remaining"] = {
                "before": pre_business_values["turns_remaining"],
                "after": stable_turns,
                "delta": -1,
            }
        if pre_hand_count and no_cards_votes == len(frames):
            business_deltas["hand"] = {"before_count": pre_hand_count, "after_count": 0}
        stable_terminal_nodes = [
            node_name
            for node_name, votes in terminal_votes.items()
            if votes == len(frames)
        ]
        if stable_terminal_nodes:
            business_deltas["terminal"] = {"nodes": stable_terminal_nodes}

        observations = tuple(
            _observe_post_submit_live_selection(context, frame) for frame in frames
        )
        valid_observations = tuple(
            observation for observation in observations if observation is not None
        )
        detection_complete = len(valid_observations) == len(frames)
        geometry_stable = detection_complete and all(
            observation[0] == valid_observations[0][0]
            for observation in valid_observations[1:]
        )
        consensus = (
            resolve_live_selected_slot_consensus(
                observation[1] for observation in valid_observations
            )
            if geometry_stable
            else None
        )
        health_votes = sum(
            _recognition_hit(context, "ProduceRecognitionHealthFlag", frame)
            for frame in frames
        )
        playable_votes = sum(
            _recognition_hit(context, "ProduceRecognitionSkipRound", frame)
            for frame in frames
        )
        secondary_votes = sum(
            _recognition_hit(context, "ProduceRecognitionChooseMoveCards", frame)
            for frame in frames
        )
        scene_start = len(scene_batch.events) if scene_batch is not None else 0
        popup_votes = sum(scene_batch.popup_hit(frame) if scene_batch is not None
                          else _recognition_hit(context, "ProduceButton", frame) for frame in frames)
        stable_unselected = bool(
            consensus and consensus.accepted and consensus.selected_slot is None
        )
        # 1967 changed the selected card instead of consuming it: the new
        # score was another card's preview, not an actual business delta.
        # Full87 lost one hand detection; missing SELECT evidence must not
        # turn that same preview into a confirmed score change either.
        unconfirmed_preview_score = None
        if not stable_unselected:
            unconfirmed_preview_score = business_deltas.pop('score', None)
        # A selected-card preview changes the stamina number before business
        # submission.  Until a dedicated preview/cost postcondition exists,
        # stamina remains telemetry and can never be the sole commit proof.
        business_committed = bool(business_deltas)
        stable_playable = health_votes == len(frames) and playable_votes == len(frames)
        selection_cleared = (
            stable_unselected
            and secondary_votes == 0
            and popup_votes == 0
            and stable_playable
        )
        if business_committed:
            reason = "business_delta_confirmed"
        elif secondary_votes:
            reason = "unsupported_secondary_decision"
        elif popup_votes:
            reason = "unsupported_confirmation_popup"
        elif not detection_complete:
            reason = "post_hand_detection_failed"
        elif not geometry_stable:
            reason = "post_hand_geometry_unstable"
        elif consensus is None or not consensus.accepted:
            reason = "post_select_consensus_rejected"
        elif consensus.selected_slot is not None:
            reason = "selected_label_remains_after_submit"
        elif selection_cleared:
            reason = "selection_state_cleared_business_unconfirmed"
        else:
            reason = "post_business_state_unstable"
        events.append(
            {
                "phase": "execution_postcondition",
                "attempt": attempt,
                "attempted_slot": selected.slot,
                "click_dispatched": False,
                "frame_ids": [frame_identifier(frame) for frame in frames],
                "candidate_boxes": [
                    None
                    if observation is None
                    else [list(box) for box in observation[0]]
                    for observation in observations
                ],
                "geometry_stable": geometry_stable,
                "select_consensus": consensus.to_dict() if consensus else None,
                "health_votes": health_votes,
                "playable_votes": playable_votes,
                "secondary_votes": secondary_votes,
                "popup_votes": popup_votes,
                **({'scene_acquisition': scene_batch.events[scene_start:]}
                   if scene_batch is not None and len(scene_batch.events) > scene_start else {}),
                "selection_state_cleared": selection_cleared,
                "pre_business_values": pre_business_values,
                "post_business_fields": list(business_observations),
                "stable_post_values": stable_post_values,
                "business_deltas": business_deltas,
                **({'unconfirmed_preview_score_delta': unconfirmed_preview_score}
                   if unconfirmed_preview_score is not None else {}),
                "no_cards_votes": no_cards_votes,
                "terminal_votes": terminal_votes,
                "business_committed": business_committed,
                "accepted": business_committed,
                "reason": reason,
            }
        )
        if business_committed:
            return True, events
        if secondary_votes or popup_votes:
            return False, events
    return False, events


def _live_event_json_default(value):
    """Serialize only the typed OCR evidence produced by the mosaic reader."""
    if isinstance(value, SimpleNamespace) and set(vars(value)) == {"text", "score", "box"}:
        return {"text": value.text, "score": value.score, "box": list(value.box)}
    raise TypeError(f"unsupported Live audit value: {type(value).__name__}")


def _publish_hif_component(context, payload):
    sink = getattr(context, 'gkh_hif_component_sink', None)
    if callable(sink):
        sink(payload)
    logger.warning(json.dumps(payload, ensure_ascii=False, default=_live_event_json_default))


def _bind_hif_opening_policy(context, event, profile='joint-search'):
    from live_decision.high_score.hif_policy_factory import bind_hif_policy

    return bind_hif_policy(context, event, profile, catalog_path=CATALOG_PATH)


def _persist_hif_mcts_computation(context, policy):
    """Called before fresh input capture or after settlement, never inside A/B."""
    from live_decision.high_score.hif_fast_native import HifFastNativePolicy
    from live_decision.high_score.hif_mcts_native import HifMctsNativePolicy

    candidate = getattr(policy, 'decision_policy', None)
    sink = getattr(context, 'gkh_hif_mcts_cache_sink', None)
    if isinstance(candidate, (HifMctsNativePolicy,HifFastNativePolicy)) and callable(sink):
        response = getattr(policy, 'response', None)
        if response is not None and not response.has_action_intent:
            # A failed computation remains unresumable. Do not replace its
            # recorded STOP reason with the export guard for incomplete work.
            return
        sink(candidate)


def _persist_hif_fast_projection(context, policy):
    from live_decision.high_score.hif_fast_native import HifFastNativePolicy
    candidate = getattr(policy,'decision_policy',None)
    if isinstance(candidate,HifFastNativePolicy):
        candidate.retain_projected_events(policy)
        _persist_hif_mcts_computation(context,policy)


def _confirm_hif_mcts_observed_commit(context, policy, event, *, submitted):
    """Called after actual input, never from prediction or preinput refresh."""
    from live_decision.high_score.hif_fast_native import HifFastNativePolicy
    from live_decision.high_score.hif_mcts_native import HifMctsNativePolicy, HifMctsSelectedIntentPolicy

    candidate = getattr(policy, 'decision_policy', None)
    if isinstance(candidate,HifFastNativePolicy):
        settlement = event.get('settlement',event)
        if submitted is not True or settlement.get('accepted') is not True:
            # Keep the original pending intent and the transaction's actual
            # failure. No observed history has been committed at this point.
            event['mcts_commit'] = dict(retained=False, reason='no_complete_observed_submitted_successor')
            return
        if settlement.get('terminal') is True:
            if getattr(context,'gkh_hif_terminal_event',None) is not settlement:
                raise ValueError('fast_terminal_requires_current_observed_result')
            candidate.confirm_terminal(policy,settlement,submitted=submitted)
            event['mcts_commit'] = dict(retained=False,terminal=True,
                reason='fast_terminal_score_observed_no_successor')
            return
        after = getattr(context,'gkh_hif_committed_state',None)
        if after is None or getattr(context,'gkh_hif_committed_event',None) is not settlement:
            raise ValueError('fast_commit_requires_current_observed_settlement')
        candidate.confirm_observed(policy,after,settlement,submitted=submitted)
        _persist_hif_mcts_computation(context,policy)
        event['mcts_commit'] = dict(retained=True,reason='fast_complete_observed_history')
        return
    if isinstance(candidate, HifMctsSelectedIntentPolicy):
        event['mcts_commit'] = dict(retained=False, reason='recovered_selected_intent_has_no_restored_tree')
        settlement = event.get('settlement', event)
        after = getattr(context, 'gkh_hif_committed_state', None)
        restore = getattr(context, 'gkh_hif_mcts_restore_selected_cache', None)
        if (submitted is True and settlement.get('accepted') is True
                and after is not None and getattr(context, 'gkh_hif_committed_event', None) is settlement
                and callable(restore)):
            event['mcts_commit'] = restore(policy, after, settlement)
        return
    if not isinstance(candidate, HifMctsNativePolicy):
        return
    settlement = event.get('settlement', event)
    after = getattr(context, 'gkh_hif_committed_state', None)
    if (submitted is not True or settlement.get('accepted') is not True
            or after is None or getattr(context, 'gkh_hif_committed_event', None) is not settlement):
        candidate.discard_pending()
        event['mcts_commit'] = dict(retained=False, reason='no_complete_observed_submitted_successor')
        return
    try:
        retrieval = getattr(context, 'gkh_hif_mcts_retrieval_commit', None)
        if retrieval is not None:
            projected, choice_action, choice_submitted = retrieval
            if projected.source_bundle != policy.bundle or projected.action != policy.response.action:
                raise ValueError('native MCTS retrieval differs from the bound drink')
            candidate.confirm_retrieval_committed(run_id=policy.bundle.run_id,
                revision=policy.bundle.revision, before=policy.bundle.state,
                action=policy.response.action, pending_state=projected.pending_state,
                choice_action=choice_action, after=after, drink_submitted=submitted,
                choice_submitted=choice_submitted, accepted=True)
        else:
            candidate.confirm_committed(run_id=policy.bundle.run_id, revision=policy.bundle.revision,
                before=policy.bundle.state, action=policy.response.action, after=after,
                submitted=True, accepted=True, drink_slots=settlement.get('drink_slots'),
                verified_successors=policy.committed_transition())
        event['mcts_commit'] = dict(retained=True, reason=('rule_maintained_submitted_successor'
            if settlement.get('rule_progress') else 'complete_observed_submitted_successor'))
        _persist_hif_mcts_computation(context, policy)
    except (ValueError, RuntimeError) as error:
        candidate.discard_pending()
        event['mcts_commit'] = dict(retained=False, reason=str(error))
    finally:
        context.gkh_hif_mcts_retrieval_commit = None


def _hif_selected_refresh_model_frames(refresh, selected_slot):
    """Reuse this selected A's complete observations, never an input permission."""
    from live_decision.hif_skill_support import _verify_selection

    try:
        event, frames = refresh['event'], refresh['frames']
        if (type(selected_slot) is not int
                or event.get('phase') != 'authoritative_refresh_a'
                or event.get('accepted') is not True
                or event.get('selected_identity_authorized') is not True
                or event.get('geometry_stable') is not True
                or event.get('click_dispatched') is not False
                or event.get('input_may_have_been_dispatched') is not False
                or type(event.get('expected_selected_slot')) is not int
                or event['expected_selected_slot'] != selected_slot
                or type(refresh.get('current_selected_slot')) is not int
                or refresh['current_selected_slot'] != selected_slot
                or not isinstance(frames, (tuple, list)) or len(frames) != 3):
            raise ValueError('selected A is not completely bound to the target')
        evidence = {key: event.get(key) for key in (
            'frame_ids', 'geometry_stable', 'select_consensus', 'select_evidence')}
        _verify_selection([frame_identifier(frame) for frame in frames], selected_slot, evidence)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise LiveSelectionError('selected_model_refresh_unresolved: ' + str(error)) from error
    return frames, evidence


def _hif_selected_schedule_support_required(bundle, receipt):
    """A known changed reservation needs its existing complete effect reader."""
    from live_decision.high_score.skill_support import supported_definition

    if receipt.get('accepted') is not True or receipt.get('single_plus_estimate') is not True:
        return False
    ordinary = bundle.kernel.cards[receipt['definition_id']]
    try:
        supported = supported_definition(ordinary, enhancement_count=receipt['plus_count'])
    except ValueError:
        return False
    return tuple(effect for effect in ordinary.effects if effect.scheduled_effects) != tuple(
        effect for effect in supported.effects if effect.scheduled_effects)


def _prepare_hif_selected_model_evidence(context, policy, selected_slot, events, *,
                                       selected_refresh=None):
    """Read model evidence before B, leaving B as the final fresh input proof.

    Ordinary and multi-plus cards need only the D-047 title/SELECT evidence.
    Exact support is retained for an explicitly bound historical receipt, the
    separate base-to-plus identity case, or three complete current header costs
    contradicting the ordinary extra-plus estimate. A known changed reservation
    also needs its full existing support proof before creating future effects.
    None of these qualifications substitutes for complete effect observations.
    An already-selected, input-free A can supply its original frames and SELECT
    votes; the title is still read independently and B is still captured fresh.
    """
    from live_decision.hif_single_plus import (
        observe_selected_single_plus,
        observe_selected_printed_cost_conflict,
    )
    from live_decision.hif_skill_support import observe_selected_skill_support

    if selected_refresh is not None:
        frames, evidence = _hif_selected_refresh_model_frames(selected_refresh, selected_slot)
        events.append(dict(phase='hif_selected_model_evidence_reuse',
            source_phase='authoritative_refresh_a', selected_slot=selected_slot,
            frame_ids=list(evidence['frame_ids']), click_dispatched=False))
    else:
        for attempt in range(2):
            frames = _capture_three_frames(context)
            try:
                consensus, _, select, geometry = _observe_live_selected_slot_consensus(
                    context, frames, expected_selected_slot=selected_slot,
                    expected_hand_count=len(policy.bundle.state.deck.hand))
            except LiveSelectionError as error:
                rejected = tuple(getattr(error, 'events', ()))
                if (attempt or len(rejected) != 1
                        or rejected[0].get('phase') != 'hif_select_label'
                        or rejected[0].get('reason') != 'exact_select_low_confidence'
                        or rejected[0].get('slot') != selected_slot
                        or rejected[0].get('input_click_count') != 0
                        or rejected[0].get('frame_id') not in {frame_identifier(frame) for frame in frames}):
                    raise
                events.append(dict(phase='hif_support_selection_reobservation', click_dispatched=False,
                    reason='exact_select_transient_low_confidence', prior=rejected,
                    frame_ids=[frame_identifier(frame) for frame in frames]))
                time.sleep(SELECTION_SETTLE_SECONDS)
                continue
            if geometry or attempt:
                break
            events.append(dict(phase='hif_support_selection_reobservation', click_dispatched=False,
                reason='selection_transition_geometry_unstable',
                frame_ids=[frame_identifier(frame) for frame in frames]))
            time.sleep(SELECTION_SETTLE_SECONDS)
        evidence = dict(frame_ids=[frame_identifier(frame) for frame in frames],
            geometry_stable=geometry, select_consensus=consensus.to_dict() if consensus else None,
            select_evidence=select)
    exact_required = getattr(policy, 'selected_skill_support', None) is not None
    if not exact_required:
        try:
            receipt = observe_selected_single_plus(context, frames, bundle=policy.bundle,
                selected_slot=selected_slot, selection_evidence=evidence, frame_identifier=frame_identifier)
        except ValueError as error:
            raise LiveSelectionError('selected_single_plus_unresolved: ' + str(error), events=[dict(
                phase='hif_selected_single_plus', accepted=False, reason=str(error),
                selection_evidence=evidence, click_dispatched=False)]) from error
        events.append(dict(phase='hif_selected_single_plus', receipt=receipt, click_dispatched=False))
        if receipt.get('accepted') is True:
            estimate = receipt if receipt.get('single_plus_estimate') is True else None
            policy.bind_selected_single_plus_estimate(estimate)
            cost_probe = observe_selected_printed_cost_conflict(context, frames,
                bundle=policy.bundle, action=policy.response.action,
                receipt=receipt, frame_identifier=frame_identifier)
            if cost_probe.get('observations'):
                events.append(dict(phase='hif_selected_printed_cost',
                    **cost_probe, click_dispatched=False))
            exact_required = (cost_probe.get('requires_exact_support') is True
                or _hif_selected_schedule_support_required(policy.bundle, receipt))
            if not exact_required:
                return None, estimate
        else:
            exact_required = receipt.get('requires_exact_support') is True
        if not exact_required:
            raise LiveSelectionError('selected_single_plus_unresolved: ' + receipt.get('reason', 'unknown'))
    receipt = observe_selected_skill_support(context, frames, bundle=policy.bundle,
        selected_slot=selected_slot, selection_evidence=evidence, frame_identifier=frame_identifier)
    events.append(dict(phase='hif_selected_skill_support', receipt=receipt, click_dispatched=False))
    if receipt.get('accepted') is not True:
        raise LiveSelectionError('selected_skill_support_unresolved: ' + receipt.get('reason', 'unknown'))
    policy.bind_selected_skill_support(receipt)
    return receipt, None


def _submitted_hif_single_plus_estimate(receipt, *, policy, submit_count, submit_events):
    """Distinguish one confirmed dispatch from the existing maybe-input count."""
    if receipt is None:
        return None
    from live_decision.hif_single_plus import verify_selected_single_plus_estimate

    action = policy.response.action
    dispatched = [row for row in submit_events if row.get('phase') == 'execution_submit'
                  and row.get('click_dispatched') is True]
    if (type(submit_count) is not int or submit_count != 1 or len(dispatched) != 1
            or dispatched[0].get('attempted_slot') != action.source_slot):
        raise LiveSelectionError('single_plus_submission_not_confirmed')
    # Never mutate the saved preinput receipt into an input permission.
    return verify_selected_single_plus_estimate(dict(receipt, submit_count=1),
        bundle=policy.bundle, action=action)


def _observe_hif_rule_progress(context, event, *, core, expected, baseline, action, preinput,
                               action_confirmation=None, inventory_hand=None, hand_slot_identity=None,
                               pending_initial_core=None, single_plus_estimate=None):
    """Bind only fresh public core to the already verified rule trajectory."""
    from live_decision.high_score.contracts import ActionKind
    from live_decision.high_score.hif_mcts_cache import json_value
    from live_decision.high_score.hif_rule_progress import with_hand_projection, resolve_rule_preinput, resolve_rule_progress

    bundle = context.gkh_hif_opening_state
    source = baseline.get('rule_preinput', baseline.get('rule_progress', baseline))
    source = json_value(source)
    source = with_hand_projection(bundle, source, getattr(context, 'gkh_hif_current_hand_projection', None))
    if pending_initial_core is not None:
        from live_decision.high_score.hif_rule_progress import resolve_rule_pending

        after, classified = resolve_rule_pending(bundle, expected, baseline_event=source,
            action=action, initial_core=pending_initial_core, core=core,
            single_plus_estimate=single_plus_estimate, hand_slot_identity=hand_slot_identity)
        event['pending_core_classification'] = classified['pending_core_classification']
        if after is None:
            event['pending_core_fallback'] = classified['reason']
            return event
        event.update(classified, force_fresh_components=False, components={},
                     observation_mode='pending_core_comparison')
        preinput = classified['pending_outcome'] == 'before'
        receipt_key = 'rule_preinput' if preinput else 'rule_progress'
        receipt = classified[receipt_key]
    elif preinput:
        after, receipt = resolve_rule_preinput(bundle, source, core)
        receipt_key = 'rule_preinput'
    else:
        after, receipt = resolve_rule_progress(bundle, expected, baseline_event=source,
            action=action, core=core, submitted=True, action_confirmation=action_confirmation,
            inventory_hand=inventory_hand,hand_slot_identity=hand_slot_identity)
        receipt_key = 'rule_progress'
    event.update({key: receipt[key] for key in ('verification_scope', 'accepted',
        'strategy_state_bound', 'run_id', 'revision', 'checks')})
    event.update(core=receipt['core'] if 'core' in receipt else json_value(core), **{receipt_key: receipt})
    if after is None:
        event['reason'] = receipt['reason']
        return event
    event.update(expected_state=receipt['expected_state'],
        observed_hand_instance_ids=[ref.instance_id for ref in after.deck.hand])
    slots = bundle.drink_slots
    if not preinput and action.kind is ActionKind.USE_DRINK:
        from live_decision.high_score.hif_rule_progress import rule_drink_slots

        slots = rule_drink_slots(bundle, action, action_confirmation)
    event['drink_slots'] = slots
    context.gkh_hif_committed_state = after
    context.gkh_hif_committed_event = event
    if not preinput:
        context.gkh_hif_opening_state = replace(bundle, drink_slots=slots)
    return event


def _hif_deterministic_completion(context, action, expected, *, observations=None):
    """Confirm card removal using layout only; no card identities or HUD OCR."""
    from live_decision.high_score.contracts import ActionKind
    from live_decision.high_score.hif_information_boundary import classify_information_boundary

    bundle = context.gkh_hif_opening_state
    boundary = classify_information_boundary(bundle.kernel, bundle.state, action, plan_type=bundle.kernel.plan_type)
    from live_decision.hif_upstream_support import supported_boundary
    boundary = supported_boundary(bundle, action, boundary,
        getattr(context, 'gkh_hif_upstream_confirmation', None))
    states = expected if isinstance(expected, tuple) else (expected,)
    if (action.kind is not ActionKind.PLAY_CARD or boundary.kind != 'deterministic_same_turn'
            or not boundary.successor.deck.hand
            or len(boundary.successor.deck.hand) != len(bundle.state.deck.hand)-1
            or any(bundle.kernel._canonical_probability_state(state) !=
                bundle.kernel._canonical_probability_state(boundary.successor) for state in states)):
        return None
    if getattr(context, 'gkh_hif_upstream_card_ui', False) is True:
        confirmation = getattr(context, 'gkh_hif_upstream_confirmation', None)
        return confirmation if confirmation and confirmation.get('response_confirmed') is True else None
    for attempt in range(4):
        frames = _capture_three_frames(context)
        observed = dict(attempt=attempt, frame_ids=[frame_identifier(frame) for frame in frames],
            expected_hand_count=len(boundary.successor.deck.hand), accepted=False)
        try:
            selection, rows, evidence, stable = _observe_live_selected_slot_consensus(
                context, frames, expected_selected_slot=None)
            observed.update(geometry_stable=stable, hand_counts=[len(row) for row in rows],
                selected_slot=selection.selected_slot if selection is not None else None,
                selection_accepted=selection.accepted if selection is not None else False,
                geometry_evidence=evidence)
            if (stable and selection is not None and selection.accepted
                    and selection.selected_slot is None
                    and all(len(row) == len(boundary.successor.deck.hand) for row in rows)):
                observed['accepted'] = True
                return dict(kind='hand_removal', frame_ids=[frame_identifier(frame) for frame in frames],
                    before_count=len(bundle.state.deck.hand), after_count=len(boundary.successor.deck.hand),
                    action_instance_id=action.card_instance_id, selected_slot=None, stable=True,
                    geometry_evidence=evidence)
        except LiveSelectionError as error:
            observed.update(reason=str(error), failure_events=list(getattr(error, 'events', ())))
        finally:
            if observations is not None:
                observations.append(observed)
        if attempt < 3:
            time.sleep(.1)
    # A dispatched card can remain selected throughout its score animation.
    # Lack of the cheap removal proof must reach the existing bounded fresh
    # public-state reader, never invent a completed transition or resubmit.
    return None


def _hif_drink_completion(context, frames, core, action):
    import cv2
    from live_decision.drink_panel import drink_slot_targets, empty_drink_slot_targets

    template = cv2.imread(str(_resolve_runtime_path(
        Path('assets/resource/base/image/produce/hif_empty_drink_glyph.png'),
        Path('resource/base/image/produce/hif_empty_drink_glyph.png'))), cv2.IMREAD_GRAYSCALE)
    readings = []
    for frame in frames:
        empty = empty_drink_slot_targets(frame, template,
            client_size=getattr(context, 'gkh_hif_client_size', None))
        readings.append((drink_slot_targets(context, frame), empty))
    if not readings or len(readings) != len(core['frame_ids']):
        raise ValueError('drink_completion_requires_observed_frames')
    if all(empty[action.source_slot] is not None and filled[action.source_slot] is None
           for filled,empty in readings):
        return dict(kind='drink_slot_empty', slot=action.source_slot, drink_id=action.drink_id,
            frame_ids=list(core['frame_ids']), empty_observations=len(readings), stable=True,
            observations=[empty[action.source_slot] for _,empty in readings])
    bundle = getattr(context,'gkh_hif_opening_state',None)
    old = tuple(getattr(bundle,'drink_slots', ()) or ())
    if (action.source_slot,action.drink_id) not in old:
        raise ValueError('consumed_drink_slot_not_observed_empty')
    after = tuple(enumerate(drink for slot,drink in old if slot != action.source_slot))
    observations = []
    for filled,empty in readings:
        row = []
        for slot in range(4):
            occupied = slot < len(after)
            retained_empty = not occupied and slot not in dict(old) and empty[slot] is None
            if ((filled[slot] is not None) != occupied
                    or occupied and empty[slot] is not None
                    or not occupied and empty[slot] is None and not retained_empty):
                raise ValueError('consumed_drink_slot_not_observed_empty')
            row.append(dict(slot=slot,state='OCCUPIED' if occupied else 'EMPTY',
                box=filled[slot] if occupied else None if retained_empty else empty[slot]['box'],
                **(dict(evidence='retained_unchanged_empty') if retained_empty else
                   {} if occupied else dict(empty_proof=empty[slot]))))
        observations.append(row)
    proof = dict(kind='drink_slots_compacted',slot=action.source_slot,drink_id=action.drink_id,
        frame_ids=list(core['frame_ids']),stable=True,observations=observations,
        before_drink_slots=old,after_drink_slots=after,
        identity_source='original_order_after_confirmed_consumption')
    from live_decision.high_score.hif_rule_progress import rule_drink_slots

    rule_drink_slots(bundle,action,proof)
    return proof


def _capture_hif_drink_completion(context, frames, core, action, *, recapture, event):
    """Only observe again; never repeat an already submitted drink."""
    from itertools import count

    poll = (getattr(context, 'gkh_hif_poll_battle_progress', None)
            if getattr(context, 'gkh_hif_upstream_card_ui', False) is True else None)
    started = time.perf_counter()
    original = core
    for attempt in (count() if poll is not None else range(2)):
        try:
            proof = _hif_drink_completion(context, frames, core, action)
            if poll is None and time.perf_counter()-started > 5:
                raise ValueError('drink_completion_observation_budget_expired')
            return core, proof
        except ValueError as error:
            if (poll is None and attempt) or str(error) != 'consumed_drink_slot_not_observed_empty':
                raise
            event.setdefault('drink_completion_reobservations', []).append(dict(
                reason=str(error), rejected_frame_ids=list(core['frame_ids']), input_click_count=0))
            if poll is not None:
                poll(frames[-1])
                time.sleep(.25)
            frames, current = recapture()
            core = {k: current[k] for k in ('fields', 'families', 'frame_ids')}
            if (any(json.dumps(core[k]) != json.dumps(original[k]) for k in ('fields', 'families'))
                    or poll is None and set(core['frame_ids']) & set(original['frame_ids'])):
                raise ValueError('drink_completion_state_changed_during_reobservation') from error
            if poll is None and time.perf_counter()-started > 5:
                raise ValueError('drink_completion_observation_budget_expired') from error
    raise AssertionError('drink completion observation exhausted')


def _observe_hif_opening_state(context, params, *, committed_expected=None, baseline_event=None,
                               committed_action=None, preinput_revalidation=False,
                               force_fresh_components=False, resumed_fresh_components=None,
                               resumed_core=None, preinput_continuity=None, single_plus_estimate=None,
                               restart_observation=None, keep_verified_card_definitions=False,
                               pending_observation=False, confirmed_choice=None, observed_battle_exit=False):
    from dataclasses import asdict

    from live_decision.inventory import capture_initial_live_checkpoint, initial_checkpoint_retry_evidence
    from live_decision.opening_observer import (
        OPENING_MODES,
        opening_component,
        partial_hud_drinks,
        opening_hand_titles,
        requested_opening_modes,
        opening_component_params,
        opening_component_retryable,
        read_opening_identity_titles,
    )
    from live_decision.high_score.kernel import KernelError
    from live_decision.high_score.opening import HifOpeningState, assemble_hif_opening, resolve_hif_committed_observation

    started = time.perf_counter()
    payloads, components = [], {}
    unreadable_live_frames = None
    displayed_result = None
    limit_batch_time = getattr(context, 'gkh_hif_upstream_card_ui', False) is not True
    previous_card_art = getattr(context, 'gkh_hif_card_art', None)
    is_post = committed_expected is not None
    single_settlement = (not limit_batch_time and is_post and committed_action is not None
        and not preinput_revalidation)
    event = dict(event='live_committed_state' if is_post else 'live_opening_state', accepted=False, executed=False,
                 execution_submit_count=0, total_input_click_count=0)
    previous_inspection = getattr(context, 'gkh_hif_current_hand_inspection', None)
    resumed_opening_inspection = (not is_post and getattr(context, 'gkh_hif_unstarted_core', None) is not None
        and isinstance(previous_inspection, dict) and previous_inspection.get('opening') is True)
    if (pending_observation or resumed_opening_inspection) and previous_inspection is not None:
        # Proven earlier information input is provenance, not input in this
        # observation. Its original receipt/count stays unchanged.
        event['retained_hand_identity_inspection'] = previous_inspection
    if preinput_revalidation:
        event['event'] = 'live_preinput_state'
        known_hand = baseline_event or getattr(context, 'gkh_hif_committed_event', None) or {}
        inspected = known_hand.get('retained_hand_identity_inspection', known_hand.get('hand_identity_inspection'))
        if inspected is not None:
            event['retained_hand_identity_inspection'] = inspected
    if force_fresh_components:
        event.update(force_fresh_components=True, maintained_components=[])
    if not is_post:
        context.gkh_hif_opening_state = None
        context.gkh_hif_terminal_event = None
    elif not preinput_revalidation:
        context.gkh_hif_committed_state = None
        context.gkh_hif_committed_event = None
    # Preinput observes the same, still-unsubmitted hand. Keep its verified
    # business baseline available to all readers until the new receipt is
    # produced. revalidate_slow_hif_decision checks that fresh receipt and
    # restores the original context on either success or failure; old input
    # coordinates and timestamps never follow this retained business state.
    try:
        run = getattr(context, 'gkh_hif_live_context', None)
        if not isinstance(run, HifLiveRunContext):
            raise ValueError('opening observation requires a verified HIF context')
        if is_post and (getattr(context, 'gkh_hif_opening_state', None) is None
                        or not isinstance(baseline_event, dict) or baseline_event.get('accepted') is not True):
            raise ValueError('HIF settlement requires the in-process native opening')
        if pending_observation and (not is_post or not force_fresh_components
                or committed_action is None or preinput_revalidation or preinput_continuity is not None):
            raise ValueError('HIF pending observation requires a complete independent post-dispatch read')
        if keep_verified_card_definitions:
            from live_decision.high_score.contracts import ActionKind
            from live_decision.inventory_customization import retained_ambiguous_card_ids

            if (keep_verified_card_definitions is not True or not is_post or force_fresh_components is not True
                    or committed_action is None or committed_action.kind is not ActionKind.END_TURN):
                raise ValueError('fixed card definitions require a fresh pending END settlement')
            candidates = committed_expected if isinstance(committed_expected, tuple) else (committed_expected,)
            for candidate in candidates:
                retained_ambiguous_card_ids(context.gkh_hif_opening_state, candidate)
            event['keep_verified_card_definitions'] = True
        if single_plus_estimate is not None:
            from live_decision.hif_single_plus import verify_selected_single_plus_estimate
            if not is_post or preinput_revalidation or preinput_continuity is not None:
                raise ValueError('HIF single-plus estimate requires post-submission settlement')
            single_plus_estimate = verify_selected_single_plus_estimate(single_plus_estimate,
                bundle=context.gkh_hif_opening_state, action=committed_action)
            event['single_plus_estimate'] = single_plus_estimate
        if is_post:
            possible = committed_expected if isinstance(committed_expected,tuple) else (committed_expected,)
            if any(state.terminal for state in possible):
                from live_decision.hif_result import observe_hif_terminal_result
                event.update(observe_hif_terminal_result(context,possible,
                    capture_frame=lambda: context.tasker.controller.post_screencap().wait().get(),
                    frame_identifier=frame_identifier,
                    round_context=getattr(context, 'gkh_hif_round_context', None)))
                if event['accepted']:
                    context.gkh_hif_terminal_event = event
                return event
            if observed_battle_exit:
                from live_decision.hif_result import try_displayed_terminal_frame

                frame = context.tasker.controller.post_screencap().wait().get()
                result = try_displayed_terminal_frame(context, frame, possible,
                    frame_identifier=frame_identifier, observed_battle_exit=True)
                if result is not None:
                    event.update(result)
                    context.gkh_hif_terminal_event = event
                    return event
        if (is_post and not pending_observation and not preinput_revalidation and committed_action is not None
                and getattr(context, 'gkh_hif_information_boundary_execution', False) is True
                and not getattr(context, 'gkh_hif_require_fresh_rule_core', False)):
            completion_observations = []
            completion = _hif_deterministic_completion(context, committed_action, committed_expected,
                observations=completion_observations)
            if completion_observations:
                event['deterministic_completion_observations'] = completion_observations
            if completion is not None:
                return _observe_hif_rule_progress(context, event, core=None, expected=committed_expected,
                    baseline=baseline_event, action=committed_action, preinput=False,
                    action_confirmation=completion)
            if completion_observations:
                event['deterministic_completion_fallback'] = 'fresh_public_state_after_unconfirmed_removal'
        recognizer = EmbeddingCardRecognizer.load(MODEL_ROOT)
        context.gkh_hif_card_art = FrozenMaskedArt.load(recognizer, MODEL_ROOT / 'hif')
        if is_post:
            possible=committed_expected if isinstance(committed_expected,tuple) else (committed_expected,)
            if possible and not pending_observation and all(not state.deck.hand for state in possible):
                context.gkh_hif_expected_empty_hand=(run.run_id,run.revision)

        initial_hand_arrival = True
        survivor_families = None
        if is_post and not preinput_revalidation and committed_action is not None:
            from live_decision.high_score.hif_rule_progress import settlement_survivor_families

            survivor_families = settlement_survivor_families(context.gkh_hif_opening_state,
                committed_expected, action=committed_action,
                baseline_event=baseline_event.get('rule_preinput', baseline_event.get('rule_progress', baseline_event)))

        def checkpoint(frames, **kwargs):
            nonlocal unreadable_live_frames
            unreadable_live_frames = None
            if len(frames) == 1:
                kwargs.setdefault('single_frame', True)
            try:
                observed = _inventory_live_checkpoint(context, frames, recognizer,
                    identity_threshold=params['debug_identity_threshold'],
                    identity_min_margin=params['debug_identity_min_margin'],
                    **(dict(settlement_survivors=survivor_families) if survivor_families is not None else {}),
                    **kwargs)
                survivors = [proof for proof in observed.get('title_family_corroboration', ())
                    if proof.get('scope') == 'rule_surviving_hand_family']
                if survivors:
                    event.setdefault('surviving_hand_observations', []).append(dict(
                        frame_ids=observed['frame_ids'], proofs=survivors))
            except ValueError as error:
                if is_post and not preinput_revalidation and frames:
                    unreadable_live_frames = frames
                identity_pending = (
                    str(error) == 'inventory navigation Live visual family is unresolved'
                    and bool(getattr(error, 'events', ()))
                    and all(e.get('phase') == 'inventory_live_identity'
                            for e in error.events))
                layout_pending = (single_settlement and str(error) == 'inventory fresh recheck is not an unselected hand'
                    or isinstance(error, LiveCardLayoutError)
                    and (str(error) == 'HIF ordinary card row is unstable'
                         or initial_hand_arrival and not preinput_revalidation
                         and str(error) == 'HIF card row dimensions are uncalibrated'))
                details = getattr(error, 'events', ())
                empty_pending = (is_post and initial_hand_arrival and not preinput_revalidation
                    and (single_settlement or not kwargs.get('single_frame')) and possible
                    and all(not state.deck.hand for state in possible)
                    and str(error) == 'empty hand needs explicit zero-card label and no card candidates'
                    and len(details) == 1
                    and details[0].get('phase') == 'inventory_empty_hand_contradiction'
                    and details[0].get('geometry_stable') is False
                    and details[0].get('candidates') == []
                    and details[0].get('frame_id') in [frame_identifier(frame) for frame in frames]
                    and isinstance(details[0].get('labels'), (tuple,list))
                    and len(details[0]['labels']) <= 1
                    and all(isinstance(row.get('text'),str)
                        and type(row.get('confidence')) in (int,float)
                        and 0 <= row['confidence'] < .95 for row in details[0]['labels']))
                if not is_post or not (identity_pending or layout_pending or empty_pending or str(error) in (
                        'HIF selection geometry invalid', 'HIF raised card has unresolved SELECT')):
                    raise
                raise LiveSelectionError(str(error), events=[dict(
                    phase='hif_settlement_hand_pending', reason=str(error),
                    original_failure_events=list(getattr(error, 'events', ())),
                    frame_ids=[frame_identifier(frame) for frame in frames])]) from error
            if is_post and not pending_observation and len(observed['families']) not in {len(s.deck.hand) for s in possible}:
                raise LiveSelectionError('HIF settlement hand transition is not complete', events=[dict(
                    phase='hif_settlement_hand_pending', observed_hand_count=len(observed['families']),
                    expected_hand_counts=sorted({len(s.deck.hand) for s in possible}),
                    frame_ids=[frame_identifier(frame) for frame in frames])])
            return observed

        initial_capture_count = 0
        settlement_poll = (getattr(context, 'gkh_hif_poll_battle_progress', None)
            if is_post and not preinput_revalidation else None)

        def capture_core_frames():
            return _capture_hif_panel_frames(context)

        def capture_initial_frames():
            nonlocal initial_capture_count, displayed_result
            if is_post:
                if settlement_poll is not None:
                    if unreadable_live_frames is not None and getattr(context, 'gkh_hif_battle_handoff', None) is not None:
                        from live_decision.hif_result import try_displayed_terminal_frame

                        displayed_result = try_displayed_terminal_frame(context, unreadable_live_frames[-1],
                            possible, frame_identifier=frame_identifier)
                        if displayed_result is not None:
                            raise ValueError('HIF battle reached its displayed result')
                    if unreadable_live_frames is not None:
                        settlement_poll(unreadable_live_frames[-1])
                    else:
                        settlement_poll()
                elif initial_capture_count >= 8:
                    raise ValueError('HIF settlement initial observation budget exhausted')
                initial_capture_count += 1
                event['settlement_initial_capture_count'] = initial_capture_count
            return capture_core_frames()

        def capture_initial_checkpoint(**options):
            while True:
                try:
                    return capture_initial_live_checkpoint(capture_initial_frames, checkpoint, event, **options)
                except ValueError as error:
                    if ((is_post or getattr(context, 'gkh_hif_upstream_card_ui', False) is True)
                            and not preinput_revalidation
                            and getattr(context, 'gkh_hif_information_boundary_execution', False)
                            and getattr(context, 'gkh_hif_pending_inventory_return', None) is None
                            and getattr(context, 'gkh_hif_item_tail_resume', None) is None):
                        import sys

                        from live_decision.hif_hand_inspection import inspect_unresolved_hand

                        if inspect_unresolved_hand(context, params, error, event, sys.modules[__name__]):
                            # The cancel owns another recognizer. Restore this
                            # reader's pair before reading the unchanged hand.
                            context.gkh_hif_card_art = FrozenMaskedArt.load(recognizer, MODEL_ROOT / 'hif')
                            continue
                    geometry = initial_checkpoint_retry_evidence(error, ())
                    if (is_post and not preinput_revalidation and options.get('max_attempts') == 1
                            and (settlement_poll is not None or initial_capture_count < 8) and geometry
                            and (all(row.get('phase') == 'inventory_live_geometry'
                                     and row.get('geometry_stable') is False for row in geometry)
                                 or all(row.get('phase') == 'inventory_live_checkpoint'
                                        and row.get('unresolved_fields') for row in geometry)
                                 or all(row.get('phase') == 'hif_settlement_hand_pending'
                                        for row in geometry))):
                        # A readable old HUD may be followed by a hand still
                        # fading into place or an unreadable resource badge.
                        # Only a complete new checkpoint can settle the action.
                        time.sleep(.25)
                        continue
                    raise

        preinput_scope = None
        restart_scope = None
        if restart_observation is not None:
            from live_decision.high_score.settled_restart import _core, _frames, _no_business_input

            bundle = context.gkh_hif_opening_state
            if (not is_post or force_fresh_components is not True or committed_action is not None
                    or preinput_revalidation or preinput_continuity is not None
                    or not isinstance(restart_observation, dict)
                    or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)
                    or len(possible) != 1 or possible[0] != bundle.state
                    or any(restart_observation.get(key) != value for key, value in
                           (('run_id', run.run_id), ('revision', run.revision)))
                    or restart_observation.get('accepted') is not True
                    or restart_observation.get('returned_unchanged') is not True):
                raise ValueError('HIF restart observation lacks unchanged current-epoch continuity')
            _no_business_input(restart_observation)
            _frames(restart_observation.get('frame_ids'))
            restart_before, restart_after = (_core(restart_observation.get(key)) for key in ('before', 'after'))
            if (set(restart_before['frame_ids']) & set(restart_after['frame_ids'])
                    or set(restart_observation['frame_ids']) &
                       (set(restart_before['frame_ids']) | set(restart_after['frame_ids']))
                    or any(json.dumps(core[key], sort_keys=True) !=
                           json.dumps(baseline_event['core'][key], sort_keys=True)
                           for core in (restart_before, restart_after) for key in ('fields', 'families'))):
                raise ValueError('HIF restart observation core changed or reuses frames')
            restart_scope = _inventory_inspected_return_scope(context, restart_after)
        if (preinput_revalidation and preinput_continuity is not None and committed_action is None
                and is_post and baseline_event.get('strategy_state_bound') is True
                and baseline_event.get('checks')
                and all(value is True for value in baseline_event['checks'].values())):
            from live_decision.high_score.maintenance import preinput_maintained_components

            bundle = context.gkh_hif_opening_state
            if (bundle.run_id, bundle.revision) != (run.run_id, run.revision):
                raise ValueError('HIF preinput observation belongs to another runtime')
            preinput_maintained_components(bundle, committed_expected, core=baseline_event['core'],
                baseline=baseline_event, proof=preinput_continuity)
            preinput_scope = _inventory_inspected_return_scope(context, baseline_event['core'])
        if restart_scope is not None:
            from live_decision.inventory import initial_checkpoint_retry_evidence

            _, before = _observe_inventory_return(context, recognizer, restart_after, event, None, {},
                validated_scope=restart_scope, capture_frames=capture_initial_frames, checkpoint=checkpoint,
                observation_key='restart_initial_observations', retry_evidence=initial_checkpoint_retry_evidence)
        elif preinput_scope is not None:
            from live_decision.inventory import initial_checkpoint_retry_evidence

            _, before = _observe_inventory_return(context, recognizer, baseline_event['core'], event, None, {},
                validated_scope=preinput_scope, capture_frames=capture_initial_frames, checkpoint=checkpoint,
                observation_key='preinput_initial_observations', retry_evidence=initial_checkpoint_retry_evidence)
        else:
            initial_frames, before = capture_initial_checkpoint(
                **(dict(max_attempts=1 if getattr(context, 'gkh_hif_information_boundary_execution', False)
                    else 8, retry_delay=.25) if is_post else {}))
        if not is_post and 'hand_identity_inspection' in event:
            inspected = event['hand_identity_inspection']['candidate_families']
            if tuple(map(tuple, before['families'])) != tuple(map(tuple, inspected)):
                raise ValueError('HIF opening hand changed after title inspection')
        event['initial_core'] = {key: before[key] for key in ('fields', 'families', 'frame_ids')}

        if not is_post:
            for attr, source in (('gkh_hif_unstarted_notebook_live', 'returned notebook'),
                                 ('gkh_hif_unstarted_entry_live', 'route entry')):
                live = getattr(context, attr, None)
                if live is not None:
                    if tuple(before['fields'][:2]) != (live['turns'], live['stamina']):
                        raise ValueError(f'HIF initial inventory differs from the {source} Live')
                    setattr(context, attr, None)

        if pending_observation and getattr(context, 'gkh_hif_information_boundary_execution', False) is True:
            # Original dispatch provenance is already restored by the caller.
            # Confirm visible stability before consulting any detail panels;
            # Automatic rereads share the host's battle progress clock.
            time.sleep(.25)
            confirmed_frames, confirmed = capture_initial_checkpoint(max_attempts=1)
            _observe_hif_rule_progress(context, event, core=confirmed, expected=committed_expected,
                baseline=baseline_event, action=committed_action, preinput=False,
                pending_initial_core=event['initial_core'], single_plus_estimate=single_plus_estimate)
            if event.get('pending_core_fallback') == 'inventory hand has no unique Live slot assignment':
                from live_decision.inventory_customization import live_slot_badge_absence

                slot_proof = live_slot_badge_absence(confirmed_frames, confirmed, run)
                event['hand_slot_identity_observation'] = slot_proof
                if slot_proof['zero_slots']:
                    _observe_hif_rule_progress(context, event, core=confirmed, expected=committed_expected,
                        baseline=baseline_event, action=committed_action, preinput=False,
                        pending_initial_core=event['initial_core'], single_plus_estimate=single_plus_estimate,
                        hand_slot_identity=slot_proof)
            if event['accepted'] is True:
                return event
            initial_frames, before = confirmed_frames, confirmed
            event['initial_core'] = {key: before[key] for key in ('fields', 'families', 'frame_ids')}

        unstarted = getattr(context, 'gkh_hif_unstarted_core', None)
        unstarted_components = {}
        if not is_post and unstarted is not None:
            from live_decision.hif_navigation_recovery import verify_unstarted_core

            event['unstarted_opening_continuity'] = verify_unstarted_core(unstarted, before)
            unstarted_components = getattr(context, 'gkh_hif_unstarted_components', {})
            pending_cards = getattr(context, 'gkh_hif_pending_inventory_return', None)
            if pending_cards is not None:
                cards = dict(pending_cards, accepted=True, returned_unchanged=True,
                    after_frame_ids=event['unstarted_opening_continuity']['fresh_frame_ids'])
                unstarted_components = dict(unstarted_components, cards=cards)
                context.gkh_hif_pending_inventory_return = None
            if unstarted_components:
                if (not set(unstarted_components) <= {key for key, _ in OPENING_MODES}
                        or any(component.get('accepted') is not True or component.get('returned_unchanged') is not True
                            or (component.get('run_id'), component.get('revision')) != (run.run_id, run.revision)
                            for component in unstarted_components.values())):
                    raise ValueError('HIF untouched opening components differ from their verified source')
                event['unstarted_retained_components'] = sorted(unstarted_components)
                context.gkh_hif_unstarted_components = None
            context.gkh_hif_unstarted_core = None

        if is_post and not pending_observation and not preinput_revalidation:
            expected_fields = {(state.turns_remaining,state.stamina,state.genki,
                                state.score,state.score_multiplier) for state in possible}
            if tuple(before['fields'][:5]) not in expected_fields:
                information_boundary = getattr(context,'gkh_hif_information_boundary_execution',False) is True
                # A readable HUD can still precede automatic card
                # effects. Continue observing during the normal progress wait.
                arrivals = [dict(accepted=False,core=event['initial_core'])]
                event['settlement_arrival_observations'] = arrivals
                matched = False
                while settlement_poll is not None or not information_boundary or initial_capture_count < 8:
                    time.sleep(.25)
                    initial_frames, arrived = capture_initial_checkpoint(max_attempts=1)
                    arrival_core = {key: arrived[key] for key in ('fields','families','frame_ids')}
                    matched = tuple(arrived['fields'][:5]) in expected_fields
                    arrivals.append(dict(accepted=matched,core=arrival_core))
                    if matched or not information_boundary:
                        break
                if not matched:
                    raise ValueError('HIF settlement public resources do not match a legal outcome')
                before = arrived
                event['initial_core'] = arrival_core

        initial_hand_arrival = False
        if is_post and not pending_observation and getattr(context, 'gkh_hif_information_boundary_execution', False) is True:
            from live_decision.high_score.contracts import ActionKind
            completion_core, confirmation = event['initial_core'], None
            if committed_action is not None and committed_action.kind is ActionKind.PLAY_CARD:
                confirmation = getattr(context, 'gkh_hif_upstream_confirmation', None)
            if committed_action is not None and committed_action.kind is ActionKind.USE_DRINK:
                completion_core, confirmation = _capture_hif_drink_completion(context, initial_frames,
                    completion_core, committed_action, recapture=lambda: capture_initial_checkpoint(max_attempts=1), event=event)
                choice = confirmed_choice or getattr(context, 'gkh_hif_confirmed_choice', None)
                if choice and committed_action.drink_id == '22':
                    target, selected = choice['action'], choice['selected_panel']
                    if (choice.get('confirmation_submitted') is not True or choice.get('choice_submit_count') != 1
                            or target.get('kind') != 'MOVE_CARD' or not target.get('card_instance_id')
                            or target['card_instance_id'] != selected.get('selected_instance_id')):
                        raise ValueError('retrieval confirmation differs from its selected card')
                    confirmation['retrieved_instance_id'] = target['card_instance_id']
            _observe_hif_rule_progress(context, event, core=completion_core,
                expected=committed_expected, baseline=baseline_event, action=committed_action,
                preinput=preinput_revalidation, action_confirmation=confirmation)
            if (not preinput_revalidation and event.get('accepted') is False
                    and event.get('reason') == 'inventory hand has no unique Live slot assignment'):
                from live_decision.inventory_customization import live_slot_badge_absence

                slot_proof = live_slot_badge_absence(initial_frames,before,run)
                event['hand_slot_identity_observation'] = slot_proof
                if slot_proof['zero_slots']:
                    _observe_hif_rule_progress(context,event,core=completion_core,expected=committed_expected,
                        baseline=baseline_event,action=committed_action,preinput=False,
                        action_confirmation=confirmation,hand_slot_identity=slot_proof)
            from live_decision.high_score.hif_rule_progress import needs_hand_inventory
            if (not preinput_revalidation and event.get('accepted') is False
                    and needs_hand_inventory(context.gkh_hif_opening_state,committed_expected,event['rule_progress'])):
                event['hand_identity_disambiguation'] = dict(event['rule_progress'])
                prior_scope = getattr(context,'gkh_hif_settlement_cards_core',None)
                context.gkh_hif_settlement_cards_core = dict(run_id=run.run_id,revision=run.revision,core=completion_core)
                try:
                    _,_,returned,navigation = _observe_hif_inventory_transaction(context,recognizer,
                        maximum_age_ms=params['max_snapshot_age_ms'],
                        identity_threshold=params['debug_identity_threshold'],
                        identity_min_margin=params['debug_identity_min_margin'])
                finally:
                    context.gkh_hif_settlement_cards_core = prior_scope
                proof = dict(version=1,run_id=run.run_id,revision=run.revision,
                    initial_core=completion_core,after_core={k:returned[k] for k in ('fields','families','frame_ids')},
                    events=navigation)
                event['inventory_hand_observation'] = proof
                _observe_hif_rule_progress(context,event,core=proof['after_core'],expected=committed_expected,
                    baseline=baseline_event,action=committed_action,preinput=False,
                    action_confirmation=confirmation,inventory_hand=proof)
            return event
        resumed = resumed_fresh_components or {}
        if resumed_fresh_components is not None or resumed_core is not None:
            if (not is_post or preinput_revalidation
                    or not isinstance(resumed, dict) or not resumed
                    or not set(resumed).issubset({'cards', 'status', 'items'})
                    or not isinstance(resumed_core, dict)
                    or any(json.dumps(before[name], sort_keys=True) !=
                           json.dumps(resumed_core.get(name), sort_keys=True)
                           for name in ('fields', 'families'))
                    or any(component.get('accepted') is not True
                           or component.get('returned_unchanged') is not True
                           or component.get('run_id') != run.run_id
                           or component.get('revision') != run.revision
                           for component in resumed.values())):
                raise ValueError('HIF resumed fresh components lack same-epoch current continuity')
            event['resumed_fresh_components'] = sorted(resumed)

        maintained = dict(unstarted_components)
        if preinput_continuity is not None:
            if (not is_post or not preinput_revalidation or force_fresh_components
                    or committed_action is not None):
                raise ValueError('HIF preinput retention cannot represent a submitted action')
            from live_decision.high_score.maintenance import preinput_maintained_components
            maintained = preinput_maintained_components(context.gkh_hif_opening_state,
                committed_expected, core=before, baseline=baseline_event, proof=preinput_continuity)
            if preinput_continuity.source_history is not None:
                from live_decision.history import observe_latest_history
                from live_decision.high_score.maintenance import bind_preinput_history

                _bind_inspected_complete_observer(context, checkpoint, preinput_scope,
                    identity_threshold=params['debug_identity_threshold'],
                    identity_min_margin=params['debug_identity_min_margin'])
                current_history = observe_latest_history(context,
                    capture_frames=capture_core_frames, checkpoint=checkpoint,
                    frame_identifier=frame_identifier,
                    round_context=getattr(context, 'gkh_hif_round_context', None))
                event['preinput_history_observation'] = current_history
                preinput_continuity = bind_preinput_history(context.gkh_hif_opening_state,
                    committed_expected, baseline=baseline_event, proof=preinput_continuity,
                    current=current_history, core=before)
                maintained = preinput_maintained_components(context.gkh_hif_opening_state,
                    committed_expected, core=before, baseline=baseline_event, proof=preinput_continuity)
            event['preinput_retained_components'] = sorted(maintained)
        elif is_post and not preinput_revalidation and not force_fresh_components:
            from live_decision.high_score.maintenance import maintained_components
            maintained = maintained_components(context.gkh_hif_opening_state, committed_expected,
                core=before, baseline=baseline_event, action=committed_action)
            if single_plus_estimate is not None:
                maintained.pop('status', None)
            event['maintained_components'] = sorted(maintained)

        class ComponentContext:
            gkh_hif_component_sink = staticmethod(payloads.append)

            def __getattr__(self, name):
                return getattr(context, name)

        child = ComponentContext()
        # Transfer this recovered prefix once, after verifying the fresh opening core.
        child.gkh_hif_item_tail_resume = None
        child.gkh_hif_opening_drink_prefix = None
        child.gkh_hif_opening_drink_handoff = None
        if not is_post and unstarted is not None:
            child.gkh_hif_item_tail_resume = getattr(context, 'gkh_hif_item_tail_resume', None)
            context.gkh_hif_item_tail_resume = None
            child.gkh_hif_opening_drink_prefix = getattr(context, 'gkh_hif_opening_drink_prefix', None)
            context.gkh_hif_opening_drink_prefix = None
        if not is_post and getattr(context, 'gkh_hif_upstream_card_ui', False) is True:
            from live_decision.hif_handoff import HifBattleHandoff

            handoff = getattr(context, 'gkh_hif_battle_handoff', None)
            if isinstance(handoff, HifBattleHandoff) and handoff.ledger is not None:
                child.gkh_hif_opening_deck_ids = tuple(card.card_id for card in handoff.ledger.cards)
                child.gkh_hif_opening_drink_handoff = handoff

        def capture_opening_cards(capture_frames, read_checkpoint, inventory_event):
            from live_decision.hif_hand_inspection import unresolved_hand, inspect_unresolved_hand

            try:
                return capture_initial_live_checkpoint(capture_frames, read_checkpoint, inventory_event)
            except LiveSelectionError as error:
                hand = unresolved_hand(error)
                if hand is None or len(hand[1]) != len(before['families']):
                    raise
                predictions = error.events[0]['predictions']
                if any(tuple(row['top_k_card_ids'][0]) != tuple(before['families'][slot])
                       for slot, row in enumerate(predictions) if slot not in hand[0]):
                    raise
                inventory_event.setdefault('initial_observations', []).append(error.events)
                if not inspect_unresolved_hand(context, params, error, event, sys.modules[__name__]):
                    raise
            # The parent already observed this hand before the component began.
            # Its values constrain continuity; navigation still needs fresh pixels.
            frames, current = capture_initial_live_checkpoint(capture_frames, read_checkpoint, inventory_event)
            inspected = event['hand_identity_inspection']['candidate_families']
            if (tuple(map(tuple, inspected)) != tuple(map(tuple, before['families']))
                    or any(current[key] != before[key] for key in ('fields', 'families'))):
                raise LiveSelectionError('HIF opening hand changed after title inspection')
            return frames, current

        # Only the status-only post-action batch may hand its final complete
        # observation directly to this caller. No receipt survives this call.
        status_return = []
        status_only = (is_post and not preinput_revalidation and not force_fresh_components
            and not resumed and restart_observation is None
            and set(maintained) == {'cards', 'items', 'drinks', 'turns'})
        child.gkh_hif_status_return_checkpoint = None

        def remember_status_return(frames, core, source):
            current = child.gkh_hif_live_context
            ids = tuple(frame_identifier(frame) for frame in frames)
            if (status_return or (current.run_id, current.revision) != (run.run_id, run.revision)
                    or len(frames) != 1
                    or ids != tuple(core.get('frame_ids', ()))
                    or ids != tuple(source.get('after_frame_ids', ()))
                    or source.get('phase') != 'status_inspection'
                    or source.get('accepted') is not True or source.get('returned_unchanged') is not True
                    or any(core.get(name) != before[name] for name in ('fields', 'families'))):
                raise ValueError('HIF status return checkpoint lacks same-transaction continuity')
            status_return.append((frames, core, source))

        # Only this already accepted, current observation supplies the baseline.
        # Child components cannot obtain this scope from predicted successors.
        child.gkh_hif_complete_observation_scope = _inventory_inspected_return_scope(context, before)
        _bind_inspected_complete_observer(context, checkpoint, child.gkh_hif_complete_observation_scope,
            identity_threshold=params['debug_identity_threshold'], identity_min_margin=params['debug_identity_min_margin'])
        if is_post:
            # The confirmed deck initializes this run only. Later observations
            # follow the bound inventory and its actual committed transitions.
            child.gkh_hif_confirmed_deck_baseline = None
            outcomes=committed_expected if isinstance(committed_expected,tuple) else (committed_expected,)
            if not outcomes or any(
                    {c.instance_id:c for c in state.deck.all_cards} !=
                    {c.instance_id:c for c in outcomes[0].deck.all_cards} for state in outcomes):
                raise ValueError('HIF outcome permanent inventories differ')
            child.gkh_hif_permanent_inventory = None
            if not force_fresh_components or keep_verified_card_definitions or pending_observation:
                from live_decision.inventory_customization import retained_ambiguous_card_ids

                try:
                    # Only unchanged old instances may be retained. Every copy
                    # in a newly generated family remains in the fresh reader.
                    retained_ambiguous_card_ids(context.gkh_hif_opening_state, outcomes[0])
                except ValueError:
                    pass  # Unsupported transitions keep the complete fresh path.
                else:
                    child.gkh_hif_permanent_inventory = (context.gkh_hif_opening_state, outcomes[0])
            child.gkh_hif_record_current_composition = keep_verified_card_definitions
        components.update(maintained)
        components.update(resumed)
        required = tuple(key for key, _ in OPENING_MODES) if is_post else HifOpeningState.required_observations
        for key, mode in requested_opening_modes(required, components):
            child.gkh_hif_opening_cards_capture = (capture_opening_cards if key == 'cards'
                and not is_post and getattr(context, 'gkh_hif_upstream_card_ui', False) is True
                and getattr(context, 'gkh_hif_information_boundary_execution', False) is True
                else capture_initial_live_checkpoint)
            child.gkh_hif_status_return_checkpoint = remember_status_return if status_only and key == 'status' else None
            if limit_batch_time and time.perf_counter()-started > 170:
                raise ValueError('HIF opening observation batch exceeded time budget')
            child.gkh_hif_settlement_cards_core = (dict(run_id=run.run_id, revision=run.revision,
                core=event['initial_core']) if key == 'cards' and is_post and not preinput_revalidation else None)
            for component_attempt in range(2):
                status_return.clear()
                count = len(payloads)
                ProduceLiveDecisionObserve().run(child, SimpleNamespace(
                    custom_action_param=json.dumps(opening_component_params(params, mode))))
                if len(payloads) != count+1:
                    raise ValueError('HIF opening component did not publish one result: '+key)
                if not is_post and key == 'drinks':
                    observed = partial_hud_drinks(payloads[-1], run)
                    if observed is not None:
                        event.setdefault('partial_components', {})[key] = observed
                        limitation = HifOpeningState.observation_limitation(key, observed)
                        if limitation:
                            event['model_input_limitation'] = limitation
                            raise ValueError('HIF opening component failed: '+key)
                        components[key] = observed
                        break
                try:
                    components[key] = opening_component(payloads[-1], key, run)
                    break
                except ValueError:
                    if (component_attempt or limit_batch_time and time.perf_counter()-started>110
                            or not opening_component_retryable(payloads[-1],key)):
                        raise
                    _, current = capture_initial_live_checkpoint(
                        capture_core_frames,checkpoint,event)
                    if any(current[name]!=before[name] for name in ('fields','families')):
                        raise ValueError('HIF changed before resuming the failed component')
                    event.setdefault('component_reobservations',[]).append(key)
        handed_off = False
        if status_only and status_return:
            returned_frames, returned_core, source = status_return[0]
            status_payload = payloads[0] if len(payloads) == 1 else {}
            current = context.gkh_hif_live_context
            if (status_payload.get('event') == 'live_status_observe'
                    and status_payload.get('accepted') is True
                    and status_payload.get('status_events') == [source]
                    and components['status'].get('accepted') is True
                    and components['status'].get('returned_unchanged') is True
                    and tuple(components['status'].get('after_frame_ids', ())) == tuple(returned_core['frame_ids'])
                    and (current.run_id, current.revision) == (run.run_id, run.revision)
                    and not any(returned_core[name] != before[name] for name in ('fields', 'families'))):
                frames, core = returned_frames, returned_core
                handed_off = True
            event['status_return_core_reused'] = handed_off
        if not handed_off:
            frames, core = capture_initial_live_checkpoint(
                capture_core_frames, checkpoint, event)
        if any(core[name] != before[name] for name in ('fields','families')):
            raise ValueError('HIF opening changed across the complete observation batch')
        if limit_batch_time and time.perf_counter()-started > 180:
            raise ValueError('HIF observation batch exceeded time budget')
        if is_post:
            resolve = resolve_hif_committed_observation
            resolution_options = dict(preinput_continuity=preinput_continuity)
            if pending_observation:
                from live_decision.high_score.opening import resolve_hif_pending_observation
                resolve, resolution_options = resolve_hif_pending_observation, {}
            baseline_components = baseline_event.get('components')
            if (baseline_components is None and pending_observation
                    and baseline_event.get('verification_scope') in (
                        'rule_maintained_information_boundary_v1', 'rule_maintained_preinput_v1')):
                # The pending comparator already supports a rule-maintained
                # predecessor without a historical panel batch.
                baseline_components = {}
            if not isinstance(baseline_components, dict):
                raise ValueError('HIF settlement has no verified component baseline')
            committed_expected,comparison = resolve(context.gkh_hif_opening_state, committed_expected,
                core=core, components=components, baseline_components=baseline_components,
                committed_action=committed_action, single_plus_estimate=single_plus_estimate,
                **resolution_options)
            event.update(comparison, run_id=run.run_id, revision=run.revision,
                core={k:core[k] for k in ('fields','families','frame_ids')},
                expected_state=asdict(committed_expected) if committed_expected is not None else None,
                strategy_state_bound=comparison['accepted'])
            if comparison['accepted']:
                by_id = {ref.instance_id: ref for ref in committed_expected.deck.hand}
                context.gkh_hif_committed_state = replace(committed_expected,
                    deck=committed_expected.deck.replace_zones(hand=tuple(
                        by_id[key] for key in comparison['observed_hand_instance_ids'])))
                context.gkh_hif_committed_event = event
                context.gkh_hif_opening_state = replace(context.gkh_hif_opening_state,
                    drink_slots=comparison.get('drink_slots'))
        else:
            from live_decision.opening_support import observe_opening_skill_support
            extended_identity = None
            if len(core['families']) in (4, 5):
                observer = getattr(context, 'gkh_hif_extended_opening_observer', None)
                if not callable(observer):
                    raise ValueError('extended_opening_requires_verified_slot_observer')
                frames, core, titles, extended_identity = observer(params, frames, core, components['cards']['cards'])
                opening_support = ()
                event['extended_hand_identity'] = extended_identity
            for identity_attempt in range(0 if extended_identity is not None else 3):
                title_frames = []
                attempt = dict(frame_ids=list(core['frame_ids']), accepted=False)
                event.setdefault('opening_identity_attempts', []).append(attempt)
                try:
                    for frame in frames:
                        title_frames.append(read_opening_identity_titles(context, frame))
                    titles = tuple(opening_hand_titles(rows) for rows in title_frames)
                    opening_support = observe_opening_skill_support(frames, title_frames,
                        run_id=run.run_id, revision=run.revision, core=core,
                        inventory_rows=components['cards']['cards'])
                    if limit_batch_time and time.perf_counter()-started >= 180:
                        raise ValueError('HIF observation batch exceeded time budget')
                    attempt['accepted'] = True
                    break
                except ValueError as error:
                    attempt['reason'] = str(error)
                    if (identity_attempt == 2 or (limit_batch_time and time.perf_counter()-started >= 180)
                            or str(error) not in {
                                'HIF opening hand title confidence is unresolved',
                                'opening support title plate is incomplete',
                                'opening support blue suffix is missing or ambiguous',
                                'opening support portrait badge is incomplete or ambiguous',
                                'opening support full title is ambiguous',
                                'opening support complete title or blue suffix is unresolved',
                            }):
                        raise
                finally:
                    attempt['title_frames'] = [[dict(text=row.text, score=row.score,
                        box=list(row.box)) for row in rows if 1090 <= row.box[1] <= 1120]
                        for rows in title_frames]
                frames = capture_core_frames()
                core = checkpoint(frames)
                if any(core[name] != before[name] for name in ('fields', 'families')):
                    raise ValueError('HIF opening changed before reobserving support identity')
                if limit_batch_time and time.perf_counter()-started >= 180:
                    raise ValueError('HIF observation batch exceeded time budget')
            event.update(hand_titles=titles, opening_skill_support=opening_support)
            confirmed_deck = getattr(context, 'gkh_hif_confirmed_deck_baseline', None)
            if confirmed_deck is not None:
                event['confirmed_deck_baseline'] = confirmed_deck
            bundle = assemble_hif_opening(run, core=core, components=components,
                hand_titles=titles, data_root=CATALOG_PATH.parent,
                round_context=getattr(context, 'gkh_hif_round_context', None),
                opening_skill_support=opening_support, confirmed_deck_baseline=confirmed_deck,
                extended_hand_identity=extended_identity)
            context.gkh_hif_opening_state = bundle
            event.update(accepted=True, run_id=run.run_id, revision=run.revision,
                core={k:core[k] for k in ('fields','families','frame_ids')},
                hand_titles=titles, state=asdict(bundle.state),
                strategy_state_bound=True, submission_ready=False)
    except _LiveInputNotDispatched as error:
        if (not is_post or preinput_revalidation or committed_action is None
                or committed_action.kind.value != 'USE_DRINK'):
            raise
        # The settled-state reader's attempted input failed before dispatch.
        # Returning its actual observation leaves the drink transaction intact.
        event.update(accepted=False, reason=str(error), error_type=type(error).__name__,
                     failed_input_dispatched=False)
    except (ValueError, KeyError, TypeError, LiveSelectionError, KernelError) as error:
        event.update(accepted=False, reason=str(error), failure_events=getattr(error, 'events', ()))
        if unreadable_live_frames is not None and getattr(context, 'gkh_hif_battle_handoff', None) is not None:
            from live_decision.hif_result import try_displayed_terminal_frame

            result = displayed_result or try_displayed_terminal_frame(context, unreadable_live_frames[-1], possible,
                frame_identifier=frame_identifier)
            if result is not None:
                event.pop('reason')
                event.update(result)
                context.gkh_hif_terminal_event = event
    finally:
        # A slow preinput refresh is nested inside an active card transaction.
        # Its private recognizer/gallery pair must not replace the outer pair
        # that will classify the fresh A/B frames after this observer returns.
        if previous_card_art is not None:
            context.gkh_hif_card_art = previous_card_art
        if is_post:
            context.gkh_hif_expected_empty_hand=None
        if is_post or resumed_opening_inspection or 'hand_identity_inspection' in event:
            context.gkh_hif_current_hand_inspection=None
        if 'rule_progress' not in event and 'rule_preinput' not in event:
            event['components'] = components
        if getattr(context, 'gkh_hif_extended_opening_identity', None) is not None and not is_post:
            event['extended_hand_identity'] = context.gkh_hif_extended_opening_identity
        event.update(total_input_click_count=(sum(p.get('total_input_click_count',0) for p in payloads)
                + event.get('hand_identity_inspection', {}).get('input_click_count', 0)
                + sum(p.get('input_click_count',0) for p in event.get('inventory_hand_observation',{}).get('events',()))
                + event.get('preinput_history_observation', {}).get('input_click_count', 0)
                + getattr(context, 'gkh_hif_extended_opening_identity', {}).get('input_click_count', 0)),
            elapsed_ms=(time.perf_counter()-started)*1000)
        _publish_hif_component(context, event)
    return event


@AgentServer.custom_action("ProduceLiveDecisionObserve")
class ProduceLiveDecisionObserve(CustomAction):
    """Evaluate one DMM Live frame and click only an allowed custom decision."""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        params: dict[str, Any] = {}
        runtime_phase = "initialization"
        selection_click_count = 0
        execution_selection_click_count = 0
        execution_submit_count = 0
        selection_events: list[dict[str, Any]] = []
        foreground_preparation_events: list[dict[str, Any]] = []
        execution_selection_events: list[dict[str, Any]] = []
        authoritative_refresh_events: list[dict[str, Any]] = []
        shadow_transaction_events: list[dict[str, Any]] = []
        resource_baseline: dict[str, Any] | None = None
        would_submit = False
        gray_preview_permit = None
        selected_skill_support = None
        single_plus_estimate = None
        card_predictions = []
        fields = {}
        try:
            try:
                params = _validate_runtime_params(_parse_params(argv.custom_action_param))
            except (json.JSONDecodeError, ValueError) as error:
                logger.warning(
                    json.dumps(
                        {
                            "event": "live_decision_stop",
                            "reason": "invalid_runtime_params",
                            "detail": str(error),
                            "foreground_preparation_events": (
                                foreground_preparation_events
                            ),
                            "executed": False,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    default=_live_event_json_default,
                    )
                )
                return False
            runtime_phase = "runtime_prechecks"
            live_mode = params["live_decision_mode"]
            from live_decision.hif_handoff import require_hif_handoff, verify_hif_handoff_opening
            battle_handoff = require_hif_handoff(context, params)
            if live_mode in {
                "selection_observe",
                "selection_reset",
                "status_observe", "status_reset", "item_status_observe", "item_status_full", "drink_observe", "drink_slots_observe", "drink_reset", "drink_slot_occupancy", "drink_inventory_observe", "turn_information_observe", "inventory_observe", "inventory_items_observe", "inventory_customization_observe", "inventory_card_state_observe", "opening_state_observe", "opening_shadow_transaction", "opening_execute",
                "shadow_transaction",
                "debug_execute",
            }:
                if not _is_process_elevated():
                    logger.warning(
                        json.dumps(
                            {
                                "event": "live_decision_stop",
                                "reason": "agent_process_not_elevated",
                                "foreground_preparation_events": (
                                    foreground_preparation_events
                                ),
                                "executed": False,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                    default=_live_event_json_default,
                        )
                    )
                    return False
            if (live_mode in {'opening_state_observe','opening_shadow_transaction','opening_execute'}
                    or isinstance(getattr(context, 'gkh_hif_live_context', None), HifLiveRunContext)):
                context = _adapt_hif_capture_context(context)
            if live_mode in {'opening_state_observe','opening_shadow_transaction','opening_execute'}:
                needs_recovery = not isinstance(getattr(context,'gkh_hif_live_context',None),HifLiveRunContext)
                bound_pid = getattr(context,'gkh_hif_context_game_pid',None)
                if bound_pid is not None or battle_handoff is not None:
                    from utils.maa_input import verify_maa_input_endpoints
                    current = verify_maa_input_endpoints(context.tasker.controller.info,
                        expected_client_size=getattr(context,'gkh_hif_client_size',(720,1280)))
                    if battle_handoff is not None and current['game']['pid'] != battle_handoff.game_pid:
                        raise ValueError('HIF handoff game process changed before recovery')
                    needs_recovery = needs_recovery or current['game']['pid'] != bound_pid
                if needs_recovery:
                    from live_decision.hif_recovery import HifRecoveryError, recover_hif_live_context
                    try:
                        recovery_kwargs = ({'round_number': battle_handoff.round_number}
                            if battle_handoff is not None else {})
                        _, recovery_event = recover_hif_live_context(context, **recovery_kwargs)
                    except HifRecoveryError as error:
                        foreground_preparation_events.append(error.event)
                        raise
                    foreground_preparation_events.append(recovery_event)
            if live_mode in {
                "selection_observe",
                "selection_reset",
                "status_observe", "status_reset", "item_status_observe", "item_status_full", "drink_observe", "drink_slots_observe", "drink_reset", "drink_slot_occupancy", "drink_inventory_observe", "turn_information_observe", "inventory_observe", "inventory_items_observe", "inventory_customization_observe", "inventory_card_state_observe", "opening_state_observe", "opening_shadow_transaction", "opening_execute",
                "shadow_transaction",
                "debug_execute",
            }:
                actual_plan = _infer_plan(context)
                if actual_plan != params["expected_plan"]:
                    logger.warning(
                        json.dumps(
                            {
                                "event": "live_decision_stop",
                                "reason": "unexpected_live_plan",
                                "expected_plan": params["expected_plan"],
                                "actual_plan": actual_plan,
                                "foreground_preparation_events": (
                                    foreground_preparation_events
                                ),
                                "executed": False,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                    default=_live_event_json_default,
                        )
                    )
                    return False
            if (
                live_mode in {"debug_execute", "opening_execute"}
                and params["debug_require_foreground_window"]
            ):
                runtime_phase = "foreground_preparation"
                (
                    foreground_prepared,
                    foreground_preparation_events,
                ) = _debug_foreground_window_preparation()
                if not foreground_prepared:
                    raise LiveSelectionError(
                        "debug_foreground_preparation_failed",
                        events=tuple(foreground_preparation_events[-1:]),
                    )
            opening_policy = None
            settlement_verified = False
            if live_mode in {"opening_state_observe", "opening_shadow_transaction", "opening_execute"}:
                previous_event = getattr(context, 'gkh_hif_committed_event', None)
                previous_state = getattr(context, 'gkh_hif_committed_state', None)
                if (live_mode != 'opening_state_observe' and isinstance(previous_event, dict)
                        and previous_event.get('accepted') is True and previous_state is not None):
                    run = context.gkh_hif_live_context
                    if (previous_event.get('run_id') != run.run_id
                            or previous_event.get('revision') != run.revision):
                        raise ValueError('HIF committed continuation belongs to another runtime')
                    # Reuse only this native context's fully verified settlement.
                    # The existing fresh A/B transaction must still prove input.
                    from live_decision.high_score.opening import (
                        verified_expired_start_drinks,
                        verified_expired_start_scores,
                        expected_permanent_inventory_keys,
                    )
                    context.gkh_hif_opening_state = replace(context.gkh_hif_opening_state,
                        state=previous_state,
                        expired_start_scores=verified_expired_start_scores(
                            context.gkh_hif_opening_state, previous_state),
                        expired_start_drinks=verified_expired_start_drinks(
                            context.gkh_hif_opening_state, previous_state),
                        inventory_keys=expected_permanent_inventory_keys(
                            context.gkh_hif_opening_state,previous_state),
                        hand_instance_ids=tuple(ref.instance_id for ref in previous_state.deck.hand))
                    opening_event = previous_event
                    params = dict(params, debug_expected_initial_hand_count=len(previous_state.deck.hand))
                else:
                    restored = getattr(context, 'gkh_hif_unsubmitted_opening_event', None)
                    if restored is None:
                        opening_event = _observe_hif_opening_state(context, params)
                    else:
                        from live_decision.inventory import capture_initial_live_checkpoint
                        from live_decision.extended_opening import verify_unsubmitted_opening

                        recognizer = EmbeddingCardRecognizer.load(MODEL_ROOT)
                        context.gkh_hif_card_art = FrozenMaskedArt.load(recognizer, MODEL_ROOT / 'hif')
                        resume = dict(source=context.gkh_hif_unsubmitted_opening_source,
                            original_observation_reused=True, input_click_count=0)
                        _, fresh_core = capture_initial_live_checkpoint(
                            lambda: _capture_hif_panel_frames(context),
                            lambda frames: _inventory_live_checkpoint(context, frames, recognizer,
                                identity_threshold=params['debug_identity_threshold'],
                                identity_min_margin=params['debug_identity_min_margin']), resume)
                        verify_unsubmitted_opening(context, restored, fresh_core)
                        resume['fresh_core'] = {k: fresh_core[k] for k in ('fields','families','frame_ids')}
                        opening_event = dict(restored, unsubmitted_opening_resume=resume)
                        context.gkh_hif_unsubmitted_opening_event = None
                        _publish_hif_component(context, opening_event)
                if opening_event['accepted'] is True:
                    verify_hif_handoff_opening(context, opening_event)
                if live_mode == 'opening_state_observe' or opening_event['accepted'] is not True:
                    return False
                params = dict(params, debug_expected_initial_hand_count=len(context.gkh_hif_opening_state.state.deck.hand))
                if getattr(context, 'gkh_hif_information_boundary_execution', False) is True:
                    # The host enters with the previous action's prestate. Keep
                    # this promoted immutable source even when a fast retained
                    # decision needs no separate slow-search refresh event.
                    context.gkh_hif_execution_source_bundle = context.gkh_hif_opening_state
                opening_policy = _bind_hif_opening_policy(context, opening_event, params.get('hif_policy_profile', 'joint-search'))
                empty_execute = live_mode == 'opening_execute' and not context.gkh_hif_opening_state.state.deck.hand
                upstream_execute = (live_mode == 'opening_execute'
                    and getattr(context, 'gkh_hif_upstream_card_ui', False) is True)
                preinput_continuity = None
                if empty_execute:
                    from live_decision.high_score.maintenance import capture_preinput_continuity
                    preinput_continuity = (capture_preinput_continuity(opening_policy.bundle, opening_event)
                        if (params.get('hif_policy_profile') in ('hif-mcts-v22-trial', 'hif-strong-reference-trial-v1')
                        and getattr(context, 'gkh_hif_information_boundary_execution', False) is not True) else None)
                    empty_response = opening_policy.decide_empty_hand()
                    selection_events.append(dict(phase='hif_candidate_precompute',
                        reason=empty_response.reason, elapsed_ms=empty_response.trace.elapsed_ms,
                        click_dispatched=False, proposal=empty_response.to_dict()))
                    _persist_hif_mcts_computation(context, opening_policy)
                if upstream_execute:
                    from live_decision.hif_upstream_policy import run_strong_card_step

                    event = run_strong_card_step(sys.modules[__name__], context, params,
                        opening_policy, opening_event)
                    if event is not None:
                        return event.get('accepted') is True
                if empty_execute or (upstream_execute and opening_policy.response is not None
                        and opening_policy.response.action is not None
                        and opening_policy.response.action.kind.value == 'USE_DRINK'):
                    from live_decision.hif_joint_execution import execute_hif_joint_action

                    recognizer=EmbeddingCardRecognizer.load(MODEL_ROOT)
                    context.gkh_hif_card_art = FrozenMaskedArt.load(recognizer, MODEL_ROOT / 'hif')
                    from live_decision.hif_joint_execution import revalidate_slow_hif_decision
                    opening_event = revalidate_slow_hif_decision(context, opening_event, opening_policy,
                        observe_complete=lambda expected, baseline: _observe_hif_opening_state(context, params,
                            committed_expected=expected, baseline_event=baseline, preinput_revalidation=True,
                            preinput_continuity=preinput_continuity))
                    runtime_phase='hif_joint_action'
                    event=execute_hif_joint_action(context,opening_event,opening_policy,
                        capture_frames=lambda: _capture_hif_panel_frames(context),
                        checkpoint=lambda frames,**kwargs:_inventory_live_checkpoint(context,frames,recognizer,
                            identity_threshold=params['debug_identity_threshold'],
                            identity_min_margin=params['debug_identity_min_margin'], **kwargs),
                        frame_identifier=frame_identifier,
                        observe_settlement=lambda expected,baseline,action:_observe_hif_opening_state(context,params,
                            committed_expected=expected,baseline_event=baseline,committed_action=action))
                    execution_submit_count=event.get('execution_submit_count',0)
                    executed=event.get('accepted') is True
                    _confirm_hif_mcts_observed_commit(context, opening_policy, event,
                        submitted=execution_submit_count == 1)
                    _publish_hif_component(context,event)
                    return event['accepted']
                live_mode = 'debug_execute' if live_mode == 'opening_execute' else 'shadow_transaction'
                params = dict(params,live_decision_mode=live_mode)
            if live_mode in {"inventory_observe", "inventory_items_observe", "inventory_customization_observe", "inventory_card_state_observe"}:
                runtime_phase = "inventory_observe"
                recognizer = EmbeddingCardRecognizer.load(MODEL_ROOT)
                context.gkh_hif_card_art = FrozenMaskedArt.load(recognizer, MODEL_ROOT / "hif")
                inventory_events = []
                retained = getattr(context,'gkh_hif_permanent_inventory',None)
                try:
                    _,_,card_checkpoint,inventory_events = _observe_hif_inventory_transaction(context,recognizer,
                        maximum_age_ms=params["max_snapshot_age_ms"],
                        identity_threshold=params["debug_identity_threshold"],
                        identity_min_margin=params["debug_identity_min_margin"],
                        full_zones=live_mode in {"inventory_observe", "inventory_card_state_observe"},
                        inspect_customizations=live_mode=="inventory_card_state_observe",
                        first_page_customizations=live_mode=="inventory_customization_observe",
                        items_overview=live_mode=="inventory_items_observe")
                    if live_mode == 'inventory_card_state_observe':
                        zone_event = inventory_events[-1]
                        presence = zone_event.get('customization_presence',[])
                        details = zone_event.get('detail_observations', [])
                        from live_decision.inventory_customization import bind_inventory_customizations
                        if retained is not None:
                            from live_decision.inventory_customization import retained_permanent_card_rows
                            permanent_cards=retained_permanent_card_rows(*retained,zone_event['composition'],
                                presence=presence, details=details)
                        elif getattr(context, 'gkh_hif_confirmed_deck_baseline', None) is not None:
                            from live_decision.inventory_customization import confirmed_deck_card_rows
                            permanent_cards = confirmed_deck_card_rows(
                                context.gkh_hif_confirmed_deck_baseline, zone_event['composition'],
                                presence=presence)
                        else:
                            permanent_cards = bind_inventory_customizations(
                                zone_event['composition'], presence, details)
                        inventory_events.append({'phase':'inventory_permanent_card_state', 'accepted':True,
                            'input_click_count':0, 'cards':permanent_cards,
                            'zone_counts':zone_event['complete_zone_counts'],
                            'before_frame_ids':next(e['before_frame_ids'] for e in inventory_events
                                if 'live_state_unchanged' in e),
                            'after_frame_ids':next(e['after_frame_ids'] for e in inventory_events
                                if 'live_state_unchanged' in e),
                            'run_id':context.gkh_hif_live_context.run_id,
                            'revision':context.gkh_hif_live_context.revision,
                            'temporary_effects_bound':False, 'strategy_state_bound':False})
                        if permanent_cards and permanent_cards[0]['source'] == 'confirmed_deck_baseline':
                            inventory_events[-1]['confirmed_deck_presence'] = presence
                        if getattr(context, 'gkh_hif_record_current_composition', False):
                            inventory_events[-1]['observed_composition'] = {
                                zone: list(ids) for zone, ids in zone_event['composition'].items()}
                except LiveSelectionError as error:
                    inventory_events = list(error.events)
                    if inventory_events:
                        inventory_events[-1].setdefault('reason',str(error))
                    inventory_events.append({'phase':'inventory_state_gate','accepted':False,
                        'input_click_count':0,'reason':str(error)})
                except ValueError as error:
                    inventory_events.append({'phase':'inventory_state_gate','accepted':False,
                        'input_click_count':0,'reason':str(error)})
                _publish_hif_component(context, {
                    "event":"live_inventory_observe","executed":False,"execution_submit_count":0,
                    "total_input_click_count":sum(e.get("input_click_count",0) for e in inventory_events),
                    "accepted":bool(inventory_events and inventory_events[-1].get("accepted")),
                    "inventory_events":inventory_events,
                    "foreground_preparation_events":foreground_preparation_events,
                })
                return False
            if live_mode in {"status_observe", "status_reset", "item_status_observe", "item_status_full", "drink_observe", "drink_slots_observe", "drink_reset", "drink_slot_occupancy", "drink_inventory_observe", "turn_information_observe"}:
                from live_decision.status_panel import StatusInspectionError, observe_status_transaction

                runtime_phase = "status_observe"
                if not isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext):
                    raise LiveSelectionError("status observation requires a verified HIF context")
                recognizer = EmbeddingCardRecognizer.load(MODEL_ROOT)
                context.gkh_hif_card_art = FrozenMaskedArt.load(recognizer, MODEL_ROOT / "hif")

                def checkpoint(frames, **kwargs):
                    if len(frames) == 1:
                        kwargs.setdefault('single_frame', True)
                    return _inventory_live_checkpoint(context, frames, recognizer,
                        identity_threshold=params["debug_identity_threshold"],
                        identity_min_margin=params["debug_identity_min_margin"], **kwargs)

                def capture_panel_frames():
                    return _capture_hif_panel_frames(context)

                _bind_inspected_complete_observer(context, checkpoint,
                    getattr(context, 'gkh_hif_complete_observation_scope', None),
                    identity_threshold=params['debug_identity_threshold'], identity_min_margin=params['debug_identity_min_margin'])

                try:
                    if live_mode == "turn_information_observe":
                        from live_decision.turn_panel import observe_turn_information_transaction
                        run = context.gkh_hif_live_context
                        status_events = observe_turn_information_transaction(context,
                            capture_frames=capture_panel_frames, checkpoint=checkpoint,
                            frame_identifier=frame_identifier, run_id=run.run_id, revision=run.revision,
                            maximum_age_ms=params["max_snapshot_age_ms"],
                            round_context=getattr(context, 'gkh_hif_round_context', None))
                    elif live_mode in {"drink_observe", "drink_slots_observe", "drink_reset", "drink_slot_occupancy", "drink_inventory_observe"}:
                        import csv

                        from live_decision.drink_panel import (
                            observe_drink_slot_occupancy,
                            observe_occupied_drink_slots,
                            observe_complete_drink_inventory,
                            observe_drink_detail_transaction,
                        )
                        drink_path = _resolve_runtime_path(Path("assets/data/p_drinks.csv"), Path("data/p_drinks.csv"))
                        drink_metadata = list(csv.DictReader(drink_path.read_text(encoding="utf-8-sig").splitlines()))
                        if live_mode in {"drink_slot_occupancy", "drink_inventory_observe"}:
                            import cv2

                            template_path = _resolve_runtime_path(Path("assets/resource/base/image/produce/hif_empty_drink_glyph.png"), Path("resource/base/image/produce/hif_empty_drink_glyph.png"))
                            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                            if live_mode == "drink_inventory_observe":
                                run = context.gkh_hif_live_context
                                detail_slot = getattr(context, 'gkh_hif_drink_detail_slot', None)
                                detail_navigate = getattr(context, 'gkh_hif_drink_detail_navigate', None)
                                icon_dir = drink_path.with_name('hif_drink_icons')
                                if not icon_dir.exists():
                                    icon_dir = PROJECT_ROOT / '.local' / 'runtime-data' / 'hif_drink_icons'
                                status_events = observe_complete_drink_inventory(context,
                                    capture_frames=lambda: (context.tasker.controller.post_screencap().wait().get(),),
                                    checkpoint=checkpoint,
                                    frame_identifier=frame_identifier, metadata=drink_metadata, template=template,
                                    plan=run.plan, run_id=run.run_id, revision=run.revision,
                                    maximum_age_ms=params["max_snapshot_age_ms"], single_frame_hud=True,
                                    default_icon_dir=icon_dir, hud_detail_slot=detail_slot, hud_detail_navigate=detail_navigate)
                            else:
                                status_events = observe_drink_slot_occupancy(context,
                                    capture_frames=capture_panel_frames, checkpoint=checkpoint, template=template, single_frame=True)
                        elif live_mode == "drink_slots_observe":
                            status_events = observe_occupied_drink_slots(context,
                                capture_frames=capture_panel_frames, checkpoint=checkpoint,
                                frame_identifier=frame_identifier, maximum_age_ms=params["max_snapshot_age_ms"],
                                metadata=drink_metadata)
                        else:
                            status_events = observe_drink_detail_transaction(context,
                                capture_frames=capture_panel_frames, checkpoint=checkpoint,
                                frame_identifier=frame_identifier, maximum_age_ms=params["max_snapshot_age_ms"],
                                names=tuple(row["name"] for row in drink_metadata),
                                restore_open_panel=live_mode == "drink_reset")
                    elif live_mode in {"item_status_observe", "item_status_full"}:
                        import csv
                        from functools import partial

                        from live_decision.item_panel import (
                            resolve_item_descriptions,
                            observe_item_description_transaction,
                        )
                        from live_decision.item_hud import observe_default_item_slots, observe_hud_first_item_inventory
                        from live_decision.high_score.contracts import PlanType
                        from live_decision.high_score.catalog_v2 import compile_item_catalog

                        item_path = _resolve_runtime_path(Path("assets/data/p_items.csv"), Path("data/p_items.csv"))
                        plan = context.gkh_hif_live_context.plan
                        metadata = list(csv.DictReader(item_path.read_text(encoding="utf-8-sig").splitlines()))
                        item_catalog = compile_item_catalog(item_path, plan_type=PlanType(plan.upper()))
                        complete_items = live_mode == "item_status_full"
                        resolved = None
                        if complete_items and getattr(context, "gkh_hif_item_tail_resume", None) is None:
                            hud_reader = partial(observe_default_item_slots, directory=item_path.with_name("hif_item_icons"),
                                                 metadata=metadata)
                            resolved, status_events = observe_hud_first_item_inventory(context, hud_reader=hud_reader,
                                capture_frames=capture_panel_frames, checkpoint=checkpoint,
                                frame_identifier=frame_identifier, maximum_age_ms=params["max_snapshot_age_ms"],
                                metadata=metadata, compiled_catalog=item_catalog, plan=plan)
                        else:
                            item_pages, status_events = observe_item_description_transaction(context,
                                capture_frames=capture_panel_frames, checkpoint=checkpoint,
                                frame_identifier=frame_identifier, maximum_age_ms=params["max_snapshot_age_ms"],
                                complete_list=complete_items)
                            if complete_items:
                                try:
                                    resolved = resolve_item_descriptions(item_pages, metadata, item_catalog, plan=plan)
                                except ValueError as error:
                                    status_events[-1].update(accepted=False, identity_count_resolution_error=str(error))
                        if resolved is not None:
                            status_events[-1]["identity_count_observation"] = dict(
                                run_id=context.gkh_hif_live_context.run_id,
                                revision=context.gkh_hif_live_context.revision,
                                after_frame_ids=status_events[-1]["after_frame_ids"],
                                catalog_source_sha256=item_catalog.source_sha256,
                                items=resolved, strategy_state_bound=False)
                    else:
                        open_status = getattr(context, 'gkh_hif_unstarted_status_panel', None)
                        _, status_events = observe_status_transaction(context,
                            capture_frames=capture_panel_frames, checkpoint=checkpoint,
                            frame_identifier=frame_identifier, maximum_age_ms=params["max_snapshot_age_ms"],
                            restore_open_panel=live_mode == "status_reset",
                            **(dict(resume_open_panel=open_status)
                               if live_mode == 'status_observe' and open_status is not None else {}),
                            return_checkpoint=(getattr(context, 'gkh_hif_status_return_checkpoint', None)
                                if live_mode == 'status_observe' else None))
                except StatusInspectionError as error:
                    status_events = error.events
                _publish_hif_component(context, {
                    "event": "live_status_observe", "executed": False, "execution_submit_count": 0,
                    "total_input_click_count": sum(event.get("input_click_count", 0) for event in status_events),
                    "accepted": bool(status_events and status_events[-1].get("accepted")),
                    "status_events": status_events,
                    "foreground_preparation_events": foreground_preparation_events,
                })
                return False
            if live_mode == "selection_reset":
                runtime_phase = "selection_reset"
                try:
                    reset_events = _cancel_hif_live_selection(context, params)
                except LiveSelectionError as error:
                    reset_events = error.events
                logger.warning(json.dumps({
                    "event": "live_selection_reset", "executed": False,
                    "execution_submit_count": 0,
                    "total_input_click_count": sum(e.get("input_click_count", 0) for e in reset_events),
                    "accepted": bool(reset_events and reset_events[-1].get("accepted")),
                    "recovery_events": reset_events,
                    "foreground_preparation_events": foreground_preparation_events,
                }, ensure_ascii=False))
                return False  # Recovery always stops; it never resumes business input.
            policy = opening_policy or WeightedPolicyProvider.load(
                POLICY_PATH,
                params["preset"],
            )
            recognizer = EmbeddingCardRecognizer.load(MODEL_ROOT)
            catalog = SkillCardFeatureCatalog.load(CATALOG_PATH)
            if isinstance(getattr(context, "gkh_hif_live_context", None), HifLiveRunContext):
                import numpy as np

                context.gkh_hif_card_art = FrozenMaskedArt.load(recognizer, MODEL_ROOT / "hif")
                recognizer.embedder.embed_tensor(np.zeros((1,3,64,64), dtype=np.float32))
            from live_decision.hif_opening_art import safety_identity_verified

            safety = DeterministicLiveSafetyFilter(
                LiveSafetyConfig(
                    provider_timeout_ms=params.get("provider_timeout_ms"),
                    max_snapshot_age_ms=params.get("max_snapshot_age_ms"),
                    minimum_identity_confidence=params.get(
                        "minimum_identity_confidence"
                    ),
                    allowed_unknown_fields=params["allowed_unknown_fields"],
                ),
                current_identity_verifier=lambda state, card: safety_identity_verified(context, state, card),
            )

            # Load OCR, detector, and native resource-evidence dependencies on
            # a disposable frame so their cold-start cost is never charged to
            # the authoritative snapshot.
            runtime_phase = "observer_prewarm"
            warm_image = context.tasker.controller.post_screencap().wait().get()
            _prewarm_live_resource_evidence_dependencies()
            _ocr_field(
                context,
                warm_image,
                "turns_remaining",
                LIVE_LAYOUT_V1["turns_remaining"][0],
                LIVE_LAYOUT_V1["turns_remaining"][1],
            )
            context.run_recognition("ProduceRecognitionCards", warm_image)

            runtime_phase = "capture"
            captured_at = time.perf_counter()
            image = context.tasker.controller.post_screencap().wait().get()
            upgrade_images = [image]
            for _ in range(2):
                time.sleep(0.08)
                upgrade_images.append(context.tasker.controller.post_screencap().wait().get())
            current_frame_id = frame_identifier(image)
            if live_mode in {"shadow_transaction", "debug_execute", "opening_shadow_transaction", "opening_execute"}:
                runtime_phase = "resource_baseline"
                selected_noise_reobserved = False
                if getattr(context, 'gkh_hif_gray_selected_source', None) is not None:
                    for baseline_attempt in range(2):
                        try:
                            resource_baseline = _observe_recovered_selected_baseline(
                                context, tuple(upgrade_images), complete_event=opening_event,
                                catalog=catalog, plan=actual_plan)
                            break
                        except LiveSelectionError as error:
                            from live_decision.opening_observer import selected_baseline_noise_reobserve_allowed

                            observations = tuple(getattr(error, 'events', ()))
                            noise_retry = (len(observations) == 1
                                and selected_baseline_noise_reobserve_allowed(observations[0], params))
                            selected_journal = getattr(context, 'gkh_hif_gray_selected_source', {}) or {}
                            target_slot = (selected_journal.get('refresh_a', {}).get('proposal', {}).get(
                                'candidate_traces', {}).get('high_score', {}).get('action', {}).get('source_slot')
                                if selected_journal.get('ordinary_safe_preview') is True
                                else selected_journal.get('permit', {}).get('slot'))
                            select_retry = (len(observations) == 1 and type(target_slot) is int
                                and observations[0].get('phase') == 'hif_select_label'
                                and (observations[0].get('reason') == 'exact_select_low_confidence'
                                     or _unresolved_select_label_diagnostic(observations[0], target_slot))
                                and observations[0].get('slot') == target_slot
                                and observations[0].get('input_click_count') == 0
                                and observations[0].get('frame_id') in {frame_identifier(f) for f in upgrade_images})
                            if (baseline_attempt or len(observations) != 1
                                    or (observations[0].get('geometry_stable') is not False
                                        and not noise_retry and not select_retry)):
                                raise
                            selected_noise_reobserved = noise_retry or select_retry
                            foreground_preparation_events.append(dict(
                                phase=('selected_baseline_select_reobservation' if select_retry
                                       else 'selected_baseline_resource_reobservation' if noise_retry
                                       else 'selected_baseline_geometry_reobservation'),
                                prior=observations[0], input_click_count=0))
                            captured_at = time.perf_counter()
                            upgrade_images = list(_capture_three_frames(context))
                            image = upgrade_images[0]
                            current_frame_id = frame_identifier(image)
                else:
                    resource_baseline = _observe_live_resource_baseline(
                        context,
                        tuple(upgrade_images),
                        expected_hand_count=params[
                            "debug_expected_initial_hand_count"
                        ],
                    )
                from live_decision.opening_observer import resource_baseline_reobserve_allowed

                selected_baseline = getattr(context, 'gkh_hif_gray_selected_source', None) is not None
                if (opening_policy is not None and not selected_noise_reobserved and selected_baseline
                        and resource_baseline_reobserve_allowed(
                        resource_baseline,params.get('debug_resource_absolute_min_confidence'),
                        selected_slot=resource_baseline.get('selected_slot') if selected_baseline else None)):
                    foreground_preparation_events.append(dict(phase='resource_baseline_reobservation',
                        prior=resource_baseline,input_click_count=0))
                    captured_at = time.perf_counter()
                    upgrade_images=list(_capture_three_frames(context))
                    image=upgrade_images[0]
                    current_frame_id=frame_identifier(image)
                    if selected_baseline:
                        resource_baseline = _observe_recovered_selected_baseline(
                            context, tuple(upgrade_images), complete_event=opening_event,
                            catalog=catalog, plan=actual_plan)
                    else:
                        resource_baseline=_observe_live_resource_baseline(context,tuple(upgrade_images),
                            expected_hand_count=params['debug_expected_initial_hand_count'])
                elif opening_policy is not None and not selected_baseline:
                    # 1619: one weak badge can be followed by one empty hand
                    # frame. Spend each existing transient category once;
                    # every attempt replaces the entire three-frame baseline.
                    retried_baseline_kinds = set()
                    upstream_baseline = getattr(context, 'gkh_hif_upstream_card_ui', False) is True
                    from live_decision.hif_upstream_policy import unselected_resource_reobserve_allowed

                    # Match the existing eight-batch upstream hand observation
                    # bound. A bad batch is discarded, never merged into a pass.
                    for _ in range(7 if upstream_baseline else 2):
                        upstream_geometry_retry = (upstream_baseline
                            and unselected_resource_reobserve_allowed(resource_baseline))
                        if not upstream_geometry_retry and not resource_baseline_reobserve_allowed(
                                resource_baseline, params.get('debug_resource_absolute_min_confidence')):
                            break
                        retry_kind = ('upstream_transient_geometry' if upstream_geometry_retry else
                                      'single_empty_geometry' if resource_baseline.get('geometry_stable') is False
                                      else 'single_weak_resource')
                        if retry_kind in retried_baseline_kinds and not upstream_geometry_retry:
                            break
                        retried_baseline_kinds.add(retry_kind)
                        foreground_preparation_events.append(dict(phase='resource_baseline_reobservation',
                            retry_kind=retry_kind, prior=resource_baseline, input_click_count=0))
                        captured_at = time.perf_counter()
                        upgrade_images = list(_capture_three_frames(context))
                        image = upgrade_images[0]
                        current_frame_id = frame_identifier(image)
                        resource_baseline = _observe_live_resource_baseline(context, tuple(upgrade_images),
                            expected_hand_count=params['debug_expected_initial_hand_count'])
                if resource_baseline.get("accepted") is not True:
                    raise LiveSelectionError(
                        "authoritative_live_resource_baseline_unresolved",
                        events=(resource_baseline,),
                    )
                baseline_threshold = params.get(
                    "debug_resource_absolute_min_confidence"
                )
                baseline_confidence = resource_baseline.get("confidence")
                if (
                    isinstance(baseline_threshold, bool)
                    or not isinstance(baseline_threshold, (int, float))
                    or not math.isfinite(float(baseline_threshold))
                    or isinstance(baseline_confidence, bool)
                    or not isinstance(baseline_confidence, (int, float))
                    or not math.isfinite(float(baseline_confidence))
                    or float(baseline_confidence) < float(baseline_threshold)
                ):
                    raise LiveSelectionError(
                        "authoritative_live_resource_baseline_confidence_unapproved",
                        events=(resource_baseline,),
                    )
            if params.get("live_decision_mode") == "debug_execute" and opening_policy is None:
                turns_remaining_field = _observe_debug_preprobe_turns_remaining(
                    context,
                    image,
                )
                if not turns_remaining_field["accepted"]:
                    logger.error(
                        json.dumps(
                            {
                                "event": "live_decision_stop",
                                "reason": "preprobe_turns_remaining_unknown",
                                "foreground_preparation_events": (
                                    foreground_preparation_events
                                ),
                                "executed": False,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                    default=_live_event_json_default,
                        )
                    )
                    return False
                fields = {"turns_remaining": turns_remaining_field}
            else:
                fields = _observe_live_fields(
                    context,
                    image,
                    commit_turn=live_mode != "shadow_transaction",
                )
            runtime_phase = "observe_cards"
            (
                card_predictions,
                current_selected_slot,
                selection_click_count,
                selection_events,
            ) = _observe_cards(
                context,
                image,
                current_frame_id,
                upgrade_images=tuple(upgrade_images),
                params=params,
                turns_remaining=int(fields["turns_remaining"].get("value") or 0),
                input_deadline=None,
                recognizer=recognizer,
                catalog=catalog,
                selected_resource_baseline=resource_baseline,
                identity_observation_started_at=captured_at,
            )
            if opening_policy is not None:
                if resource_baseline.get('selected_geometry_authorized') is True:
                    transaction = resource_baseline.get('resource_transaction', {})
                    if (transaction.get('accepted') is not True
                            or transaction.get('selected_slot') != current_selected_slot
                            or current_frame_id not in transaction.get('frame_ids', ())):
                        raise LiveSelectionError('selected_recovery_initial_resources_unbound')
                    for name in ('stamina', 'shield'):
                        observed = fields[name]
                        if (observed.get('accepted') is not True
                                or observed.get('value') != transaction.get('preview_'+name)):
                            raise LiveSelectionError('selected_recovery_initial_resources_changed')
                        fields[name] = dict(accepted=True,
                            value=transaction['normalized_'+name],
                            confidence=min(float(observed.get('confidence', 0)),
                                           float(transaction['confidence'])),
                            reason='selected_preview_normalized_to_preinput_baseline',
                            preview_value=transaction['preview_'+name],
                            preview_delta=transaction[name+'_delta'])
                    fields['resource_transaction'] = transaction
                    preview = transaction.get('card_effect_preview') or {}
                    if fields['score'].get('value') != opening_policy.bundle.state.score:
                        exact_preview = (fields['score'].get('accepted') is True
                            and preview.get('run_id') == opening_policy.bundle.run_id
                            and preview.get('revision') == opening_policy.bundle.revision
                            and preview.get('selected_slot') == current_selected_slot
                            and fields['score']['value'] == preview.get('preview_score'))
                        if not exact_preview:
                            from live_decision.hif_gray_selection import ordinary_selected_observed_score_preview

                            try:
                                selected_live = DmmLiveVisualStateProvider().build_state(dict(
                                    snapshot_id=current_frame_id,capture_age_ms=0,
                                    fields=fields,card_predictions=card_predictions))
                                observed_preview = ordinary_selected_observed_score_preview(
                                    opening_policy.bundle,complete_event=opening_event,
                                    journal=getattr(context,'gkh_hif_gray_selected_source',None) or {},
                                    live=selected_live,fields=fields)
                            except ValueError as error:
                                raise LiveSelectionError('selected_recovery_initial_score_changed', events=(dict(
                                    phase='selected_recovery_initial_score', reason=str(error),
                                    score=fields.get('score'), action_order=fields.get('action_order'),
                                    card_effect_preview=preview),)) from error
                            selection_events.append(dict(phase='selected_recovery_observed_score_preview',
                                accepted=True,**observed_preview))
                        fields['score'] = dict(fields['score'],
                            value=opening_policy.bundle.state.score,
                            preview_value=fields['score']['value'],
                            reason='selected_preview_normalized_to_preinput_baseline')
                # Candidate calculation uses the complete native opening;
                # fresh A/B still bind every field and authorize every input.
                preliminary = DmmLiveVisualStateProvider().build_state(dict(
                    snapshot_id=current_frame_id,capture_age_ms=(time.perf_counter()-captured_at)*1000,
                    fields=fields,card_predictions=card_predictions))
                if hasattr(opening_policy, 'bind_visible_hand'):
                    context.gkh_hif_opening_state = opening_policy.bind_visible_hand(
                        preliminary, identity_events=selection_events)
                    if opening_policy.ordinary_plus_projection is not None:
                        context.gkh_hif_current_hand_projection = opening_policy.ordinary_plus_projection
                        _publish_hif_component(context, opening_policy.ordinary_plus_projection)
                from live_decision.high_score.maintenance import capture_preinput_continuity
                preinput_continuity = (capture_preinput_continuity(opening_policy.bundle, opening_event)
                    if (params.get('hif_policy_profile') in ('hif-mcts-v22-trial', 'hif-strong-reference-trial-v1')
                        and getattr(context, 'gkh_hif_information_boundary_execution', False) is not True) else None)
                candidate = opening_policy.decide(opening_policy.prepare_state(preliminary,fields))
                selection_events.append(dict(phase='hif_candidate_precompute',
                    reason=candidate.reason,elapsed_ms=candidate.elapsed_ms,click_dispatched=False,
                    proposal=candidate.to_dict()))
                _persist_hif_mcts_computation(context, opening_policy)
                recovery_journal = getattr(context, 'gkh_hif_gray_selected_source', None)
                recovered_support = resource_baseline.get('selected_skill_support')
                support_source = 'fresh_selected_resource_baseline'
                if recovered_support is None and isinstance(recovery_journal, dict):
                    recovered_support = recovery_journal.get('selected_skill_support')
                    support_source = 'original_unsubmitted_selection_receipt'
                if (resource_baseline.get('selected_geometry_authorized') is True
                        and isinstance(recovery_journal, dict)
                        and recovered_support is not None):
                    opening_policy.bind_selected_skill_support(recovered_support)
                    selection_events.append(dict(phase='selected_support_restored_for_preview',
                        accepted=True, source=support_source,
                        submission_authorized=False))
                if (resource_baseline.get('selected_geometry_authorized') is True
                        and isinstance(recovery_journal, dict)
                        and recovery_journal.get('single_plus_estimate') is not None):
                    opening_policy.bind_selected_single_plus_estimate(recovery_journal['single_plus_estimate'])
                    selection_events.append(dict(phase='selected_single_plus_restored_for_preview',
                        accepted=True, source='original_unsubmitted_selection_receipt',
                        submission_authorized=False))
                if live_mode == 'debug_execute':
                    from live_decision.hif_joint_execution import revalidate_slow_hif_decision
                    opening_event = revalidate_slow_hif_decision(context, opening_event, opening_policy,
                        observe_complete=lambda expected, baseline: _observe_hif_opening_state(context, params,
                            committed_expected=expected, baseline_event=baseline, preinput_revalidation=True,
                            preinput_continuity=preinput_continuity))
                joint_action = opening_policy.response.action if opening_policy.response else None
                if joint_action is not None and joint_action.kind.value in {'USE_DRINK', 'END_TURN'}:
                    if live_mode != 'debug_execute':
                        _publish_hif_component(context, dict(event='live_joint_shadow', accepted=True,
                            executed=False, execution_submit_count=0, proposal=opening_policy.response.to_dict()))
                        return False
                    from live_decision.hif_joint_execution import execute_hif_joint_action
                    runtime_phase='hif_joint_action'
                    event = execute_hif_joint_action(context, opening_event, opening_policy,
                        capture_frames=lambda: _capture_hif_panel_frames(context),
                        checkpoint=lambda frames, **kwargs: _inventory_live_checkpoint(context, frames, recognizer,
                            identity_threshold=params['debug_identity_threshold'],
                            identity_min_margin=params['debug_identity_min_margin'], **kwargs),
                        frame_identifier=frame_identifier,
                        observe_settlement=lambda expected, baseline, action: _observe_hif_opening_state(
                            context, params, committed_expected=expected, baseline_event=baseline,
                            committed_action=action))
                    execution_submit_count=event.get('execution_submit_count',0)
                    executed=event.get('accepted') is True
                    _confirm_hif_mcts_observed_commit(context, opening_policy, event,
                        submitted=execution_submit_count == 1)
                    _publish_hif_component(context, event)
                    return event['accepted']
                if live_mode == 'debug_execute' and (
                        candidate.action is not None
                        or (resource_baseline.get('selected_geometry_authorized') is True
                            and joint_action is not None and joint_action.kind.value == 'PLAY_CARD')):
                    # Calculate the completed outcome before either fresh
                    # input gate. A non-deterministic continuation is outside
                    # this first transaction's settlement observer.
                    committed_expected = opening_policy.committed_transition()
                    _persist_hif_fast_projection(context, opening_policy)
            recovered_selected = bool(
                getattr(context, 'gkh_hif_gray_selected_source', None) is not None
                and resource_baseline.get('selected_geometry_authorized') is True
                and type(current_selected_slot) is int
                and resource_baseline.get('selected_slot') == current_selected_slot
                and selection_click_count == 0)
            if live_mode == "debug_execute" and (
                (current_selected_slot is not None and not recovered_selected) or selection_click_count != 0
            ):
                raise LiveSelectionError(
                    "debug_execute_initial_state_not_unselected",
                    events=tuple(selection_events),
                )
            runtime_phase = "policy_and_safety"
            executed = False
            execution_selection_click_count = 0
            execution_submit_count = 0
            input_click_counts = (
                selection_click_count,
                execution_selection_click_count,
                execution_submit_count,
            )
            allow_input_free_snapshot_retry = all(
                type(count) is int and count == 0 for count in input_click_counts
            )
            input_free_snapshot_retries_remaining = (
                1 if allow_input_free_snapshot_retry else 0
            )
            snapshot_retry_input_click_counts = (
                input_click_counts if allow_input_free_snapshot_retry else None
            )
            if live_mode == "shadow_transaction":
                identity_certificate = _build_live_identity_certificate(card_predictions)
                runtime_phase = "authoritative_refresh_a"
                refresh_a = _capture_verified_live_refresh(
                    context,
                    identity_certificate,
                    expected_selected_slot=None,
                    expected_generation=None,
                    params=params,
                    plan=actual_plan,
                    phase="authoritative_refresh_a",
                    recognizer=recognizer,
                    catalog=catalog,
                    policy=policy,
                    safety=safety,
                    expected_hand_count=params[
                        "debug_expected_initial_hand_count"
                    ],
                    resource_baseline=resource_baseline,
                    allow_input_free_snapshot_retry=(
                        allow_input_free_snapshot_retry
                    ),
                    snapshot_retries_remaining=(
                        input_free_snapshot_retries_remaining
                    ),
                    input_click_counts=snapshot_retry_input_click_counts,
                )
                authoritative_refresh_events.append(refresh_a["event"])
                fields = refresh_a["fields"]
                card_predictions = refresh_a["card_predictions"]
                state = refresh_a["state"]
                proposal = refresh_a["proposal"]
                decision = refresh_a["decision"]
                current_selected_slot = refresh_a["current_selected_slot"]
                target_slot = (
                    decision.action.card_slot if decision.has_action_intent else None
                )
                refresh_a_accepted = bool(
                    target_slot is not None and current_selected_slot is None
                )
                shadow_transaction_events.append(
                    {
                        "phase": "shadow_refresh_a_gate",
                        "click_dispatched": False,
                        "selected_slot": current_selected_slot,
                        "target_slot": target_slot,
                        "safe_decision_has_action": decision.has_action_intent,
                        "accepted": refresh_a_accepted,
                        "reason": (
                            "authoritative_a_unselected_target_ready"
                            if refresh_a_accepted
                            else "authoritative_a_has_no_safe_unselected_target"
                        ),
                    }
                )
                if refresh_a_accepted:
                    runtime_phase = "shadow_target_selection"
                    (
                        selection_ready,
                        execution_selection_click_count,
                        target_selection_events,
                    ) = _select_live_target_for_refresh(
                        context,
                        refresh_a["candidates"],
                        current_selected_slot=None,
                        target_slot=target_slot,
                        input_deadline=refresh_a["input_deadline"],
                        authorization_floor=refresh_a["authorization_floor"],
                        unselected_transition_authorized=(
                            refresh_a["event"].get(
                                "unselected_geometry_authorized"
                            )
                            is True
                        ),
                    )
                    execution_selection_events.extend(target_selection_events)
                    shadow_transaction_events.append(
                        {
                            "phase": "shadow_target_selection",
                            "click_dispatched": bool(
                                execution_selection_click_count == 1
                            ),
                            "target_slot": target_slot,
                            "accepted": bool(
                                selection_ready
                                and execution_selection_click_count == 1
                            ),
                            "reason": (
                                "shadow_target_selection_dispatched"
                                if selection_ready
                                and execution_selection_click_count == 1
                                else "shadow_target_selection_failed"
                            ),
                        }
                    )
                    if selection_ready and execution_selection_click_count == 1:
                        runtime_phase = "authoritative_refresh_b"
                        refresh_b = _capture_selected_live_refresh(
                            context,
                            identity_certificate,
                            expected_selected_slot=target_slot,
                            expected_generation=refresh_a["generation"],
                            params=params,
                            plan=actual_plan,
                            phase="authoritative_refresh_b",
                            recognizer=recognizer,
                            catalog=catalog,
                            policy=policy,
                            safety=safety,
                            expected_hand_count=params[
                                "debug_expected_initial_hand_count"
                            ],
                            resource_baseline=resource_baseline,
                            selected_transition_verified=True,
                            allow_selected_score_preview_change=True,
                        )
                        authoritative_refresh_events.append(refresh_b["event"])
                        fields = refresh_b["fields"]
                        card_predictions = refresh_b["card_predictions"]
                        state = refresh_b["state"]
                        proposal = refresh_b["proposal"]
                        decision = refresh_b["decision"]
                        current_selected_slot = refresh_b["current_selected_slot"]
                        selected_card_present = any(
                            card.slot == target_slot and card.box is not None
                            for card in state.hand
                        )
                        would_submit = bool(
                            decision.has_action_intent
                            and decision.action.card_slot == target_slot
                            and current_selected_slot == target_slot
                            and selected_card_present
                        )
                        shadow_transaction_events.append(
                            {
                                "phase": "shadow_refresh_b_result",
                                "click_dispatched": False,
                                "selected_slot": current_selected_slot,
                                "target_slot": target_slot,
                                "resource_transaction": refresh_b.get(
                                    "resource_transaction"
                                ),
                                "safe_decision_has_action": (
                                    decision.has_action_intent
                                ),
                                "would_submit": would_submit,
                                "accepted": would_submit,
                                "reason": (
                                    "would_submit"
                                    if would_submit
                                    else "authoritative_b_redecision_changed"
                                ),
                            }
                        )
                runtime_phase = "shadow_complete"
            elif live_mode == "debug_execute":
                identity_certificate = _build_live_identity_certificate(card_predictions)
                runtime_phase = "authoritative_refresh_a"
                refresh_a = (_capture_selected_live_refresh if recovered_selected
                             else _capture_verified_live_refresh)(
                    context,
                    identity_certificate,
                    expected_selected_slot=current_selected_slot,
                    expected_generation=None,
                    params=params,
                    plan=actual_plan,
                    phase="authoritative_refresh_a",
                    recognizer=recognizer,
                    catalog=catalog,
                    policy=policy,
                    safety=safety,
                    expected_hand_count=params[
                        "debug_expected_initial_hand_count"
                    ],
                    resource_baseline=resource_baseline,
                    allow_input_free_snapshot_retry=(
                        allow_input_free_snapshot_retry
                    ),
                    snapshot_retries_remaining=(
                        input_free_snapshot_retries_remaining
                    ),
                    input_click_counts=snapshot_retry_input_click_counts,
                )
                authoritative_refresh_events.append(refresh_a["event"])
                fields = refresh_a["fields"]
                card_predictions = refresh_a["card_predictions"]
                state = refresh_a["state"]
                proposal = refresh_a["proposal"]
                decision = refresh_a["decision"]
                current_selected_slot = refresh_a["current_selected_slot"]
                if (not decision.has_action_intent and opening_policy is not None
                        and opening_policy.response.action is not None
                        and getattr(context, 'gkh_hif_gray_preview_only', False) is True):
                    from live_decision.hif_gray_selection import authorize_gray_singleton_preview
                    journal = getattr(context, 'gkh_hif_gray_selection_journal', None)
                    if not callable(journal):
                        raise LiveSelectionError('gray preview requires a durable selection journal')
                    gray_preview_permit = authorize_gray_singleton_preview(
                        context.gkh_hif_opening_state, state, complete_event=opening_event,
                        candidates=refresh_a['candidates'], identity_certificate=identity_certificate,
                        refresh_event=refresh_a['event'], planned_action=opening_policy.response.action,
                        safety=safety, prior_journal=getattr(context, 'gkh_hif_gray_selection_record', None))
                    journal('selection_reserved', dict(permit=gray_preview_permit,
                        refresh_a=refresh_a['event'], identity_certificate=identity_certificate))
                    context.gkh_hif_gray_selection_record = gray_preview_permit
                elif decision.has_action_intent and getattr(context, 'gkh_hif_gray_preview_only', False) is True:
                    journal = getattr(context, 'gkh_hif_gray_selection_journal', None)
                    if not callable(journal):
                        raise ValueError('preview-only selection requires a durable journal')
                    journal('selection_reserved', dict(ordinary_safe_preview=True,
                        refresh_a=refresh_a['event'], identity_certificate=identity_certificate))
                if decision.has_action_intent or gray_preview_permit is not None:
                    target_slot = decision.action.card_slot if decision.has_action_intent else gray_preview_permit['slot']
                    runtime_phase = "target_selection"
                    (
                        selection_ready,
                        execution_selection_click_count,
                        target_selection_events,
                    ) = _select_live_target_for_refresh(
                        context,
                        refresh_a["candidates"],
                        current_selected_slot=current_selected_slot,
                        target_slot=target_slot,
                        input_deadline=refresh_a["input_deadline"],
                        authorization_floor=refresh_a["authorization_floor"],
                        unselected_transition_authorized=(
                            refresh_a["event"].get(
                                "unselected_geometry_authorized"
                            )
                            is True
                        ),
                        require_foreground_window=params[
                            "debug_require_foreground_window"
                        ],
                    )
                    execution_selection_events.extend(target_selection_events)
                    if getattr(context, 'gkh_hif_gray_preview_only', False) is True:
                        journal('selection_dispatched', dict(selection_ready=selection_ready,
                            selection_count=execution_selection_click_count, selection_events=target_selection_events))
                    if selection_ready:
                        selected_transition_verified_for_b = bool(
                            execution_selection_click_count == 1
                            or (
                                current_selected_slot == target_slot
                                and refresh_a["event"].get(
                                    "selected_identity_authorized"
                                )
                                is True
                            )
                        )
                        if opening_policy is not None:
                            runtime_phase = 'selected_single_plus_prepare'
                            selected_skill_support, single_plus_estimate = _prepare_hif_selected_model_evidence(
                                context, opening_policy, target_slot, execution_selection_events,
                                selected_refresh=(refresh_a if (
                                    type(execution_selection_click_count) is int
                                    and execution_selection_click_count == 0
                                    and current_selected_slot == target_slot
                                    and refresh_a['event'].get('accepted') is True
                                    and refresh_a['event'].get('selected_identity_authorized') is True
                                ) else None))
                            if (selected_skill_support is not None
                                    or (single_plus_estimate or {}).get('ordinary_plus_projection') is True):
                                committed_expected = opening_policy.committed_transition()
                                _persist_hif_fast_projection(context, opening_policy)
                        runtime_phase = "authoritative_refresh_b"
                        # Only the actually dispatched ordinary selection can
                        # supply this in-process identity source. The existing
                        # carry verifier checks A, the bound state, dispatch,
                        # current overlap/SELECT and the fresh family again.
                        selected_identity_journal = (dict(
                            ordinary_safe_preview=True, refresh_a=refresh_a['event'],
                            identity_certificate=identity_certificate,
                            selection_ready=selection_ready,
                            selection_count=execution_selection_click_count,
                            business_submit_count=execution_submit_count,
                            selection_events=execution_selection_events,
                        ) if opening_policy is not None and gray_preview_permit is None
                            and execution_selection_click_count == 1 else None)
                        def capture_refresh_b(*, final_preparation_retry=False):
                            refresh_reader = (_capture_verified_live_refresh if final_preparation_retry
                                              else _capture_selected_live_refresh)
                            return refresh_reader(
                                context,
                                identity_certificate,
                                expected_selected_slot=target_slot,
                                expected_generation=refresh_a["generation"],
                                params=params,
                                plan=actual_plan,
                                phase="authoritative_refresh_b",
                                recognizer=recognizer,
                                catalog=catalog,
                                policy=policy,
                                safety=safety,
                                expected_hand_count=params[
                                    "debug_expected_initial_hand_count"
                                ],
                                resource_baseline=resource_baseline,
                                selected_transition_verified=(
                                    selected_transition_verified_for_b
                                ),
                                selected_identity_journal=selected_identity_journal,
                                allow_selected_score_preview_change=(
                                    execution_selection_click_count == 1
                                ),
                                input_click_counts=(selection_click_count,
                                    execution_selection_click_count, execution_submit_count),
                                **(dict(select_retries_remaining=0, hand_count_retries_remaining=0)
                                   if final_preparation_retry else {}),
                            )
                        refresh_b = capture_refresh_b()
                        authoritative_refresh_events.append(refresh_b["event"])
                        fields = refresh_b["fields"]
                        card_predictions = refresh_b["card_predictions"]
                        state = refresh_b["state"]
                        proposal = refresh_b["proposal"]
                        decision = refresh_b["decision"]
                        current_selected_slot = refresh_b["current_selected_slot"]
                        if getattr(context, 'gkh_hif_gray_preview_only', False) is True:
                            from dataclasses import asdict
                            journal('selected_observation', dict(refresh_b=refresh_b['event'],
                                live_state=asdict(state), fields=fields,
                                candidates=[asdict(candidate) for candidate in refresh_b['candidates']],
                                detail_identity=_hif_selected_detail_identity(context, refresh_b['frames'],
                                    tuple(identity_certificate[target_slot]['family']), catalog, actual_plan),
                                selected_slot=current_selected_slot, business_submit_count=0,
                                single_plus_estimate=single_plus_estimate,
                                selected_skill_support=selected_skill_support))
                            execution_selection_events.append(dict(phase='gray_preview_only', accepted=True,
                                click_dispatched=False, submission_authorized=False))
                        if (
                            gray_preview_permit is None
                            and not getattr(context, 'gkh_hif_gray_preview_only', False)
                            and ((decision.has_action_intent
                                  and decision.action.card_slot == target_slot
                                  and current_selected_slot == target_slot)
                                 or _unsubmitted_selected_b_stale(refresh_b, target_slot=target_slot,
                                     expected_generation=refresh_a['generation'],
                                     recovered_selected=recovered_selected,
                                     selection_events=execution_selection_events,
                                     input_click_counts=(selection_click_count,
                                         execution_selection_click_count, execution_submit_count)))
                        ):
                            runtime_phase = "execute"
                            selected = next(
                                (
                                    card
                                    for card in state.hand
                                    if card.slot == target_slot and card.box is not None
                                ),
                                None,
                            )
                            if selected is not None:
                                (
                                    executed,
                                    execution_submit_count,
                                    submit_events,
                                    refresh_b,
                                ) = _submit_selected_live_card_with_b_retry(
                                    context, refresh_b, target_slot=target_slot,
                                    opening_policy=opening_policy,
                                    expected_generation=refresh_a["generation"],
                                    capture_again=lambda: capture_refresh_b(final_preparation_retry=True),
                                    refresh_events=authoritative_refresh_events,
                                    recovered_selected=recovered_selected,
                                    selection_events=execution_selection_events,
                                    input_click_counts=(selection_click_count,
                                        execution_selection_click_count, execution_submit_count),
                                )
                                fields = refresh_b["fields"]
                                card_predictions = refresh_b["card_predictions"]
                                state = refresh_b["state"]
                                proposal = refresh_b["proposal"]
                                decision = refresh_b["decision"]
                                current_selected_slot = refresh_b["current_selected_slot"]
                                execution_selection_events.extend(submit_events)
                                if execution_submit_count == 1 and opening_policy is not None:
                                    runtime_phase = 'hif_complete_settlement'
                                    single_plus_estimate = _submitted_hif_single_plus_estimate(
                                        single_plus_estimate, policy=opening_policy,
                                        submit_count=execution_submit_count, submit_events=submit_events)
                                    if single_plus_estimate is not None:
                                        execution_selection_events.append(dict(phase='hif_single_plus_submitted',
                                            receipt=single_plus_estimate, click_dispatched=False))
                                    if getattr(opening_policy, 'observed_normal_score_preview', None) is not None:
                                        committed_expected = opening_policy.committed_transition()
                                        _persist_hif_fast_projection(context, opening_policy)
                                    settlement = _observe_hif_opening_state(context, params,
                                        committed_expected=committed_expected, baseline_event=opening_event,
                                        committed_action=opening_policy.response.action,
                                        single_plus_estimate=single_plus_estimate)
                                    settlement_verified = settlement['accepted'] is True
                                    executed = settlement_verified
                                    _confirm_hif_mcts_observed_commit(context, opening_policy, settlement,
                                        submitted=execution_submit_count == 1)
                                    execution_selection_events.append(dict(phase='hif_complete_settlement',
                                        accepted=settlement['accepted'], click_dispatched=False,
                                        reason=settlement.get('reason'),
                                        observation_input_click_count=settlement['total_input_click_count']))
                            runtime_phase = "post_execute"
                        else:
                            execution_selection_events.append(
                                {
                                    "phase": "execution_redecision_stop",
                                    "attempted_slot": target_slot,
                                    "click_dispatched": False,
                                    "reason": ("selected_input_evidence_stale" if decision.reason == "stale_state"
                                               else "authoritative_redecision_changed"),
                                    "input_may_have_been_dispatched": False,
                                }
                            )
            else:
                observation = {
                    "snapshot_id": current_frame_id,
                    "capture_age_ms": (time.perf_counter() - captured_at) * 1000.0,
                    "fields": fields,
                    "card_predictions": card_predictions,
                }
                state = DmmLiveVisualStateProvider().build_state(observation)
                proposal = policy.decide(state)
                decision = safety.filter(state, proposal)
            execution_click_count = (
                execution_selection_click_count + execution_submit_count
            )
            total_input_click_count = selection_click_count + execution_click_count
            if live_mode == "shadow_transaction" and not (
                selection_click_count == 0
                and execution_selection_click_count in {0, 1}
                and execution_submit_count == 0
                and total_input_click_count == execution_selection_click_count
            ):
                raise LiveSelectionError("shadow_transaction_input_budget_exceeded")
            decision = replace(decision, executed=executed)
            runtime_phase = "logging"
            logger.warning(
                json.dumps(
                    {
                        "event": "live_decision_observe",
                        "protocol_version": PROTOCOL_VERSION,
                        "provider": params.get("provider", "weighted-policy"),
                        "preset": params.get("preset", "conservative-basic-v1"),
                        "mode": live_mode,
                        "state_provider": DmmLiveVisualStateProvider.provider_name,
                        "snapshot_id": state.snapshot_id,
                        "capture_age_ms": state.capture_age_ms,
                        "current_selected_slot": current_selected_slot,
                        "selection_click_count": selection_click_count,
                        "identity_probe_click_count": selection_click_count,
                        "selection_events": selection_events,
                        "execution_click_count": execution_click_count,
                        "total_input_click_count": total_input_click_count,
                        "execution_selection_click_count": (
                            execution_selection_click_count
                        ),
                        "execution_submit_count": execution_submit_count,
                        "would_submit": would_submit,
                        "resource_baseline": resource_baseline,
                        "shadow_transaction_events": shadow_transaction_events,
                        "foreground_preparation_events": (
                            foreground_preparation_events
                        ),
                        "authoritative_refresh_events": authoritative_refresh_events,
                        "execution_selection_events": execution_selection_events,
                        "selected_skill_support": selected_skill_support,
                        "single_plus_estimate": single_plus_estimate,
                        "visual_fields": fields,
                        "card_predictions": [
                            {
                                **{key: value for key, value in entry.items() if key != "prediction"},
                                "prediction": entry["prediction"].to_dict(),
                            }
                            for entry in card_predictions
                        ],
                        "unknown_fields": state.unknown_fields,
                        "contradictory_fields": state.contradictory_fields,
                        "hand": [card.to_dict() for card in state.hand],
                        "proposal": proposal.to_dict(),
                        "safe_decision": decision.to_dict(),
                        "executed": executed,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_live_event_json_default,
                )
            )
            runtime_phase = "complete"
            return executed and (opening_policy is None or settlement_verified)
        except LiveSelectionError as error:
            input_may_have_been_dispatched = bool(
                selection_click_count
                or execution_selection_click_count
                or execution_submit_count
                or (
                    params.get("live_decision_mode")
                    in {"selection_observe", "debug_execute"}
                    and runtime_phase == "observe_cards"
                )
            )
            logger.error(
                json.dumps(
                    {
                        "event": "live_selection_stop",
                        "reason": str(error),
                        "runtime_phase": runtime_phase,
                        "input_may_have_been_dispatched": (
                            input_may_have_been_dispatched
                        ),
                        "identity_probe_events": selection_events,
                        "foreground_preparation_events": (
                            foreground_preparation_events
                        ),
                        "authoritative_refresh_events": authoritative_refresh_events,
                        "shadow_transaction_events": shadow_transaction_events,
                        "execution_selection_events": execution_selection_events,
                        "selected_skill_support": selected_skill_support,
                        "single_plus_estimate": single_plus_estimate,
                        "failure_events": list(error.events),
                        "identity_probe_click_count": selection_click_count,
                        "execution_selection_click_count": (
                            execution_selection_click_count
                        ),
                        "execution_submit_count": execution_submit_count,
                        "would_submit": would_submit,
                        "executed": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_live_event_json_default,
                )
            )
            return False
        except Exception as error:
            submit_dispatched = any(
                event.get("phase") == "execution_submit"
                and event.get("click_dispatched") is True
                for event in execution_selection_events
            ) or (runtime_phase == 'hif_joint_action' and execution_submit_count > 0)
            input_may_have_been_dispatched = bool(
                selection_click_count
                or execution_selection_click_count
                or execution_submit_count
                or (
                    params.get("live_decision_mode")
                    in {"selection_observe", "debug_execute"}
                    and runtime_phase == "observe_cards"
                )
            )
            logger.exception(
                json.dumps(
                    {
                        "event": "live_decision_exception_stop",
                        "reason": "unexpected_runtime_exception",
                        "error_type": type(error).__name__,
                        "runtime_phase": runtime_phase,
                        "submit_dispatched": submit_dispatched,
                        "input_may_have_been_dispatched": input_may_have_been_dispatched,
                        "identity_probe_events": selection_events,
                        "visual_fields": fields,
                        "card_predictions": [dict(entry, prediction=entry['prediction'].to_dict())
                                             for entry in card_predictions],
                        "foreground_preparation_events": (
                            foreground_preparation_events
                        ),
                        "execution_selection_events": execution_selection_events,
                        "selected_skill_support": selected_skill_support,
                        "single_plus_estimate": single_plus_estimate,
                        "authoritative_refresh_events": authoritative_refresh_events,
                        "shadow_transaction_events": shadow_transaction_events,
                        "identity_probe_click_count": selection_click_count,
                        "execution_selection_click_count": (
                            execution_selection_click_count
                        ),
                        "execution_submit_count": execution_submit_count,
                        "would_submit": would_submit,
                        "executed": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_live_event_json_default,
                )
            )
            return False
