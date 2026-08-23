"""Fail-closed Maa bridge for the DMM Live decision preview."""

from __future__ import annotations

import re
import json
import time
from typing import Any
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
    infer_dmm_max_stamina,
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
from card_selection.model import frame_identifier, isolate_card_candidates
from maa.agent.agent_server import AgentServer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = PROJECT_ROOT / "assets" / "resource" / "base" / "model" / "classify" / "card_embedding"
POLICY_PATH = PROJECT_ROOT / "assets" / "data" / "live_decision_presets.yaml"
CATALOG_PATH = PROJECT_ROOT / "assets" / "data" / "skill_cards.csv"
LIVE_CARD_ART_BOXES = (
    (53, 899, 180, 180),
    (265, 919, 190, 160),
    (497, 919, 170, 160),
)

LIVE_LAYOUT_V1 = {
    "turns_remaining": ([35, 40, 75, 90], r"^\d{1,2}$"),
    "stamina": ([555, 185, 105, 70], r"^\d{1,3}$"),
    "score_multiplier": ([100, 55, 145, 95], r"^\d{1,4}%$"),
}
LIVE_FIELD_LIMITS = {
    "turns_remaining": (1, 12),
    "stamina": (0, 100),
    "score": (0, 999999999),
    "score_multiplier": (0.01, 99.99),
}
TURN_TRACKER = DmmTurnTracker()
SELECTION_SETTLE_SECONDS = 0.35
SELECTION_FRAME_INTERVAL_SECONDS = 0.08


def _parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Live decision parameters must be a JSON object")
    return parsed


def _ocr_field(
    context: Context,
    image: Any,
    name: str,
    roi: list[int],
    expected: str,
) -> dict[str, Any]:
    node_name = f"Task200Visual{name}"
    detail = context.run_recognition(
        node_name,
        image,
        pipeline_override={
            node_name: {
                "recognition": "OCR",
                "roi": roi,
                "expected": expected,
            }
        },
    )
    results = list(getattr(detail, "filtered_results", ()) or ()) if detail and detail.hit else []
    if not results:
        return {"accepted": False, "value": None, "confidence": 0.0}
    best = max(results, key=lambda item: float(getattr(item, "score", 0.0)))
    text = str(getattr(best, "text", "")).strip().replace(",", "")
    if re.fullmatch(expected, text) is None:
        return {"accepted": False, "value": None, "confidence": 0.0}
    if name == "score_multiplier":
        value: int | float = int(text[:-1]) / 100.0
    else:
        value = int(text)
    limits = LIVE_FIELD_LIMITS.get(name)
    if limits is None:
        return {"accepted": False, "value": None, "confidence": 0.0, "text": text}
    lower, upper = limits
    if not lower <= value <= upper:
        return {"accepted": False, "value": None, "confidence": 0.0, "text": text}
    return {
        "accepted": True,
        "value": value,
        "confidence": float(getattr(best, "score", 0.0)),
        "text": text,
        "box": list(getattr(best, "box", ())),
    }


def _infer_plan(context: Context) -> str:
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


def _capture_three_frames(context: Context) -> tuple[Any, Any, Any]:
    frames = []
    for index in range(3):
        frames.append(context.tasker.controller.post_screencap().wait().get())
        if index < 2:
            time.sleep(SELECTION_FRAME_INTERVAL_SECONDS)
    return tuple(frames)


def _locate_selected_live_slots(
    context: Context,
    image: Any,
    candidates: tuple[Any, ...],
) -> tuple[int, ...]:
    """Locate SELECT labels only; yellow recommendation borders are ignored."""

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


def _click_live_candidate(context: Context, candidate: Any) -> bool:
    x, y, width, height = candidate.box
    return bool(
        context.tasker.controller.post_click(
            x + width // 2,
            y + height // 2,
        ).wait().succeeded
    )


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


def _observe_cards(
    context: Context,
    image: Any,
    frame_id: str,
    upgrade_images: tuple[Any, Any, Any],
    *,
    params: dict[str, Any],
    turns_remaining: int,
) -> tuple[list[dict[str, Any]], int | None, int, list[dict[str, Any]]]:
    detail = context.run_recognition("ProduceRecognitionCards", image)
    results = list(getattr(detail, "all_results", ()) or ()) if detail and detail.hit else []
    candidates = normalise_live_candidates(isolate_card_candidates(results))
    if not candidates:
        return [], None, 0, []
    recognizer = EmbeddingCardRecognizer.load(MODEL_ROOT)
    catalog = SkillCardFeatureCatalog.load(CATALOG_PATH)
    plan = _infer_plan(context)
    identity_threshold = float(params.get("debug_identity_threshold", 0.70))
    identity_min_margin = float(params.get("debug_identity_min_margin", 0.0))
    try:
        predictions = classify_live_card_views(
            recognizer,
            image,
            candidates,
            frame_id=frame_id,
            plan=plan,
            acceptance_threshold=identity_threshold,
            min_margin=identity_min_margin,
            legacy_art_boxes=LIVE_CARD_ART_BOXES if len(candidates) == 3 else None,
        )
    except LiveCardLayoutError:
        return [], None, 0, []

    initial_selection = resolve_live_selected_slot_consensus(
        _locate_selected_live_slots(context, frame, candidates) for frame in upgrade_images
    )
    selection_events: list[dict[str, Any]] = [
        {
            "phase": "initial_observation",
            "attempted_slot": None,
            "click_dispatched": False,
            "frame_ids": [frame_identifier(frame) for frame in upgrade_images],
            "select_consensus": initial_selection.to_dict(),
        }
    ]
    if not initial_selection.accepted:
        raise LiveSelectionError(
            "initial three-frame Live SELECT state is ambiguous",
            events=selection_events,
        )
    current_selected_slot = initial_selection.selected_slot
    title_views = {
        view.slot: view
        for view in build_live_title_views(
            tuple(image.shape),
            candidates,
            selected_slot=current_selected_slot,
        )
    }
    upgrades: dict[int, LiveUpgradeConsensus] = {}
    requires_upgrade: dict[int, bool] = {}
    family_ids_by_slot: dict[int, tuple[str, ...]] = {}
    for candidate, prediction in zip(candidates, predictions, strict=True):
        family_ids = prediction.top_k_card_ids[0] if prediction.top_k_card_ids else ()
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
                upgrade_images,
                title_views[candidate.slot],
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
    unresolved = [
        candidate
        for candidate, prediction in zip(candidates, predictions, strict=True)
        if prediction.accepted
        and requires_upgrade[candidate.slot]
        and not upgrades[candidate.slot].accepted
    ]
    selectable_unresolved = (
        unresolved
        if len(candidates) in {4, 5}
        and params.get("live_decision_mode") == "debug_execute"
        else ()
    )
    for candidate in selectable_unresolved:
        if current_selected_slot == candidate.slot:
            raise LiveSelectionError(
                "refusing to click the currently selected Live card",
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
        selected_frames = _capture_three_frames(context)
        selection_consensus = resolve_live_selected_slot_consensus(
            _locate_selected_live_slots(context, frame, candidates)
            for frame in selected_frames
        )
        selection_event = {
            "phase": "selection_observation",
            "attempted_slot": candidate.slot,
            "click_dispatched": True,
            "frame_ids": [frame_identifier(frame) for frame in selected_frames],
            "select_consensus": selection_consensus.to_dict(),
        }
        selection_events.append(selection_event)
        if (
            not selection_consensus.accepted
            or selection_consensus.selected_slot != candidate.slot
        ):
            raise LiveSelectionError(
                "three-frame Live SELECT evidence did not match the clicked slot",
                events=selection_events,
            )
        current_selected_slot = candidate.slot
        selected_view = build_live_title_views(
            tuple(selected_frames[-1].shape),
            candidates,
            selected_slot=current_selected_slot,
        )[current_selected_slot]
        upgrade = resolve_live_title_consensus(selected_frames, selected_view)
        selection_event["upgrade_consensus"] = upgrade.to_dict()
        if not upgrade.accepted:
            raise LiveSelectionError(
                "selected Live title did not resolve enhancement",
                events=selection_events,
            )
        upgrades[candidate.slot] = upgrade

    observed: list[dict[str, Any]] = []
    for candidate, prediction in zip(candidates, predictions, strict=True):
        upgrade = upgrades[candidate.slot]
        family_ids = family_ids_by_slot[candidate.slot]
        eligible_ids = _eligible_family_ids(catalog, family_ids, plan)
        if prediction.accepted and len(eligible_ids) == 1:
            resolved_id = eligible_ids[0]
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
                    "identity:visual-family",
                    f"enhancement:{enhancement_tag}",
                ],
                "upgraded": entry.upgraded if entry is not None else None,
                "upgrade_evidence": upgrade.to_dict(),
            }
        )
    return observed, current_selected_slot, selection_click_count, selection_events


def _execute_selected_live_card(
    context: Context,
    selected: Any,
    hand: tuple[Any, ...],
    current_selected_slot: int | None,
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
    executed = _click_live_candidate(context, selected)
    click_count += 1
    return executed and click_count == required_clicks, click_count, selection_events


@AgentServer.custom_action("ProduceLiveDecisionObserve")
class ProduceLiveDecisionObserve(CustomAction):
    """Evaluate one DMM Live frame and click only an allowed custom decision."""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            params = _parse_params(argv.custom_action_param)
            if params.get("live_decision_mode") not in {"observe", "debug_execute"}:
                reason = "unsupported_live_decision_mode"
                logger.warning(
                    json.dumps(
                        {"event": "live_decision_stop", "reason": reason, "executed": False},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                return False

            captured_at = time.perf_counter()
            image = context.tasker.controller.post_screencap().wait().get()
            upgrade_images = [image]
            for _ in range(2):
                time.sleep(0.08)
                upgrade_images.append(context.tasker.controller.post_screencap().wait().get())
            current_frame_id = frame_identifier(image)
            fields = {
                name: _ocr_field(context, image, name, roi, expected)
                for name, (roi, expected) in LIVE_LAYOUT_V1.items()
            }
            fields["turn"] = (
                TURN_TRACKER.observe(fields["turns_remaining"].get("value"), time.monotonic())
                if fields["turns_remaining"]["accepted"]
                else {"accepted": False, "value": None, "confidence": 0.0}
            )
            fields["max_stamina"] = (
                infer_dmm_max_stamina(image, fields["stamina"].get("value"))
                if fields["stamina"]["accepted"]
                else {"accepted": False, "value": None, "confidence": 0.0}
            )
            fields["action_order"] = locate_dmm_self_rank(image)
            if fields["action_order"]["accepted"]:
                fields["score"] = _ocr_field(
                    context,
                    image,
                    "score",
                    fields["action_order"]["score_roi"],
                    r"^\d+$",
                )
            else:
                fields["score"] = {"accepted": False, "value": None, "confidence": 0.0}
            for unresolved in ("items", "field_effects", "statuses"):
                fields[unresolved] = {"accepted": False, "value": None, "confidence": 0.0}
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
            )
            observation = {
                "snapshot_id": current_frame_id,
                "capture_age_ms": (time.perf_counter() - captured_at) * 1000.0,
                "fields": fields,
                "card_predictions": card_predictions,
            }
            state = DmmLiveVisualStateProvider().build_state(observation)
            policy = WeightedPolicyProvider.load(
                POLICY_PATH,
                str(params.get("preset", "conservative-basic-v1")),
            )
            proposal = policy.decide(state)
            safety = DeterministicLiveSafetyFilter(
                LiveSafetyConfig(
                    provider_timeout_ms=params.get("provider_timeout_ms"),
                    max_snapshot_age_ms=params.get("max_snapshot_age_ms"),
                    minimum_identity_confidence=params.get("minimum_identity_confidence"),
                    allowed_unknown_fields=tuple(
                        str(value) for value in params.get("allowed_unknown_fields", ())
                    ),
                )
            )
            decision = safety.filter(state, proposal)
            executed = False
            execution_click_count = 0
            execution_selection_events: list[dict[str, Any]] = []
            if params.get("live_decision_mode") == "debug_execute" and decision.has_action_intent:
                selected = next(
                    (card for card in state.hand if card.slot == decision.action.card_slot),
                    None,
                )
                if selected is not None and selected.box is not None:
                    selectable_hand = tuple(card for card in state.hand if card.box is not None)
                    (
                        executed,
                        execution_click_count,
                        execution_selection_events,
                    ) = _execute_selected_live_card(
                        context,
                        selected,
                        selectable_hand,
                        current_selected_slot,
                    )
            logger.warning(
                json.dumps(
                    {
                        "event": "live_decision_observe",
                        "protocol_version": PROTOCOL_VERSION,
                        "provider": params.get("provider", "weighted-policy"),
                        "preset": params.get("preset", "conservative-basic-v1"),
                        "state_provider": DmmLiveVisualStateProvider.provider_name,
                        "snapshot_id": state.snapshot_id,
                        "current_selected_slot": current_selected_slot,
                        "selection_click_count": selection_click_count,
                        "selection_events": selection_events,
                        "execution_click_count": execution_click_count,
                        "execution_selection_events": execution_selection_events,
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
                )
            )
            return executed
        except LiveSelectionError as error:
            logger.error(
                json.dumps(
                    {
                        "event": "live_selection_stop",
                        "reason": str(error),
                        "selection_events": list(error.events),
                        "executed": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return False
        except Exception:
            logger.exception("Live 只读视觉状态提供者失败，已在出牌前停止")
            return False
