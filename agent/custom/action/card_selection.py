from __future__ import annotations

import json
import time
from typing import Any
from pathlib import Path
from collections import defaultdict

from utils import logger
from maa.context import Context
from card_selection import (
    MultiFrameVoter,
    SelectionContext,
    CaptureEnvironment,
    CardCaptureAdapter,
    EmbeddingCardRecognizer,
    load_policy,
    validate_dmm_window,
    new_capture_batch_id,
)
from utils.local_data import StorageSafetyError, get_runtime_store
from maa.custom_action import CustomAction
from card_selection.model import frame_identifier, isolate_card_candidates
from card_selection.types import UNKNOWN, CandidateBox, CardPrediction
from maa.agent.agent_server import AgentServer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = PROJECT_ROOT / "assets" / "resource" / "base" / "model" / "classify" / "card_embedding"
POLICY_PATH = PROJECT_ROOT / "assets" / "data" / "card_selection_presets.yaml"


def _parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("card-selection parameters must be a JSON object")
    return parsed


def _action_params(node: Any) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    return node.get("action", {}).get("param", {}).get("custom_action_param", {})


def _recognition_params(node: Any) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    return node.get("recognition", {}).get("param", {}).get("custom_recognition_param", {})


def _infer_context(context: Context, params: dict[str, Any]) -> SelectionContext:
    effect = _action_params(context.get_node_data("ProduceChooseNIAEventFlag")).get("effect", "")
    inferred_plan = "free"
    if str(effect).startswith("感性"):
        inferred_plan = "sense"
    elif str(effect).startswith("理性"):
        inferred_plan = "logic"
    elif str(effect).startswith("非凡"):
        inferred_plan = "anomaly"

    idol = _recognition_params(context.get_node_data("ProduceChooseIdol"))
    idol_key = "/".join(str(idol.get(key, "unknown")) for key in ("idol_name", "song_name"))
    scenario_node = context.get_node_data("ProduceChooseScenario") or {}
    scenario = scenario_node.get("recognition", {}).get("param", {}).get("template", [])
    scenario_text = " ".join(scenario if isinstance(scenario, list) else [str(scenario)]).lower()
    inferred_scene = "NIA" if "nia" in scenario_text else "初"
    deck = params.get("deck_card_ids", [])
    return SelectionContext(
        scene=str(params.get("scene", inferred_scene)),
        plan=str(params.get("plan", inferred_plan)),
        p_idol=str(params.get("p_idol", idol_key)),
        stage=str(params.get("stage", "unknown")),
        deck_card_ids=tuple(str(value) for value in deck),
    )


def _roi_rejection(
    candidate: CandidateBox,
    frame_id: str,
    error: Exception,
    recognizer: EmbeddingCardRecognizer,
) -> CardPrediction:
    return CardPrediction(
        slot=candidate.slot,
        box=candidate.box,
        frame_id=frame_id,
        card_id=UNKNOWN,
        predicted_card_id=UNKNOWN,
        class_name="",
        confidence=0.0,
        margin=0.0,
        raw_score=0.0,
        accepted=False,
        reason=f"invalid_roi:{type(error).__name__}",
        model_sha256=recognizer.model_sha256,
        classes_sha256=recognizer.classes_sha256,
        detector_label=candidate.detector_label,
        detector_score=candidate.detector_score,
        source_frames=(frame_id,),
        recognizer=recognizer.NAME,
        gallery_sha256=recognizer.gallery_sha256,
    )


@AgentServer.custom_action("ProduceChooseCardIdObserve")
class ProduceChooseCardIdObserve(CustomAction):
    """Compute a safe reward-card decision, log it, and deliberately never click.

    The only GUI-exposed mode is ``observe``. Returning ``False`` is deliberate:
    the current task must stop at the read-only G2 gate instead of falling back to
    a recommendation or first-slot click.
    """

    MAX_FRAMES = 3
    EXPECTED_CANDIDATES = 3

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            params = _parse_params(argv.custom_action_param)
            if params.get("mode", "observe") != "observe":
                logger.error("卡牌 ID 选择器拒绝非只读模式；G3 执行器尚未授权接入")
                return False
            manifest = json.loads((MODEL_ROOT / "manifest.json").read_text(encoding="utf-8"))
            runtime = manifest["runtime"]
            threshold = params.get("acceptance_threshold", runtime["acceptance_threshold"])
            minimum_margin = params.get("minimum_margin", runtime["minimum_margin"])
            threshold_value = float(threshold) if threshold is not None else None
            margin_value = float(minimum_margin) if minimum_margin is not None else None
            recognizer = EmbeddingCardRecognizer.load(MODEL_ROOT)
            policy = load_policy(POLICY_PATH, str(params.get("preset", "conservative-basic-v1")))
            selection_context = _infer_context(context, params)
            store = get_runtime_store()
            environment = None
            batch_id = None
            if store.config.enabled:
                controller_info = context.tasker.controller.info
                if not isinstance(controller_info, dict) or controller_info.get("type") != "win32":
                    logger.error("卡牌采集拒绝非 PC/Win32 控制器；请切换到 DMM-PC 配置")
                    return False
                window_facts = validate_dmm_window(controller_info.get("hwnd"))
                resource = context.tasker.resource
                if not resource.loaded:
                    logger.error("卡牌采集拒绝未完成加载的资源")
                    return False
                sentinel = context.get_node_data("Task100DmmResourceSentinel") or {}
                if (
                    sentinel.get("recognition", {}).get("type") != "DirectHit"
                    or sentinel.get("action", {}).get("type") != "DoNothing"
                ):
                    logger.error("卡牌采集未检测到 DMM 资源哨兵；拒绝将官服/模拟器画面标记为 DMM")
                    return False
                environment = CaptureEnvironment.dmm_runtime(
                    resource_hash=resource.hash,
                    window_class=window_facts.window_class,
                    process_name=window_facts.process_name,
                    client_size=window_facts.client_size,
                )
                batch_id = new_capture_batch_id()
            capture = CardCaptureAdapter(
                store,
                batch_id=batch_id,
                requested_role="reward_card_runtime",
                source_type="dmm_pc",
                environment=environment,
            )
            observations: dict[int, list[CardPrediction]] = defaultdict(list)
            final_predictions: tuple[CardPrediction, ...] | None = None
            for capture_index in range(self.MAX_FRAMES):
                image = context.tasker.controller.post_screencap().wait().get()
                current_frame_id = frame_identifier(image, capture_index)
                detail = context.run_recognition("ProduceRecognitionCards", image)
                results = (detail.all_results or []) if detail and detail.hit else []
                candidates = isolate_card_candidates(results)
                if len(candidates) != self.EXPECTED_CANDIDATES:
                    logger.error(
                        json.dumps(
                            {
                                "event": "card_selection_stop",
                                "reason": "candidate_count_mismatch",
                                "frame_id": current_frame_id,
                                "expected": self.EXPECTED_CANDIDATES,
                                "actual": len(candidates),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    try:
                        capture.capture_failure_frame(
                            image,
                            frame_id=current_frame_id,
                            capture_index=capture_index,
                            reason="candidate_count_mismatch",
                            candidate_count=len(candidates),
                        )
                    except StorageSafetyError:
                        logger.exception("卡牌失败现场写入失败，采集器已暂停；本次不重试写入")
                    return False
                frame_predictions: list[CardPrediction] = []
                for candidate in candidates:
                    try:
                        prediction = recognizer.classify(
                            image,
                            candidate,
                            frame_id=current_frame_id,
                            plan=selection_context.plan,
                            acceptance_threshold=threshold_value,
                            min_margin=margin_value,
                        )
                    except ValueError as error:
                        prediction = _roi_rejection(candidate, current_frame_id, error, recognizer)
                    observations[candidate.slot].append(prediction)
                    frame_predictions.append(prediction)
                try:
                    capture.capture_normal_frame(
                        image,
                        frame_id=current_frame_id,
                        capture_index=capture_index,
                        candidates=candidates,
                        predictions=frame_predictions,
                    )
                except StorageSafetyError:
                    logger.exception("卡牌画面写入失败，采集器已暂停；本次不重试写入")
                    return False
                if all(prediction.accepted for prediction in frame_predictions):
                    final_predictions = tuple(frame_predictions)
                    break
                if capture_index + 1 < self.MAX_FRAMES:
                    time.sleep(0.25)
            if final_predictions is None:
                voter = MultiFrameVoter(
                    minimum_frames=2,
                    minimum_agreeing=2,
                    acceptance_threshold=threshold_value,
                    minimum_margin=margin_value,
                )
                final_predictions = tuple(voter.vote(observations[slot]) for slot in sorted(observations))
            decision = policy.decide(final_predictions, selection_context)
            logger.info(
                json.dumps(
                    {
                        "event": "card_selection_decision",
                        "mode": "observe",
                        "executed": False,
                        "context": selection_context.to_dict(),
                        "decision": decision.to_dict(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            if decision.has_click_intent:
                logger.warning("只读门已生成拟选结果，但未执行点击；等待 G2 人工复核与后续 G3 单批授权")
            else:
                logger.warning(f"卡牌选择在点击前安全停止: {decision.reason}")
        except Exception:
            logger.exception("卡牌 ID 只读选择器失败，已在点击前停止")
        return False
