"""Read-only HIF page candidates; uncalibrated card/attribute pages stay blocked."""

from __future__ import annotations

import re

from .hif import day_from_countdown


def _results(context, name, image):
    result = context.run_recognition(name, image)
    return tuple(getattr(result, "filtered_results", ()) or ()) if result and result.hit else ()


def observe_hif_scene(context, image) -> dict:
    """No controller, screenshot capture, input or identity inference.

    Recognition geometry is a hiro-derived candidate awaiting frozen-frame
    calibration. A successful page candidate alone cannot enable input.
    """
    if getattr(image, "shape", ())[:2] != (1280, 720):
        return {"scene": "unknown", "reason": "unsupported_capture_resolution"}
    if _results(context, "ProduceHifBattleText", image) or (
        _results(context, "ProduceRecognitionHealthFlag", image)
        and _results(context, "ProduceRecognitionSkipRound", image)
    ):
        return {"scene": "battle_entry", "reason": "battle_requires_task275"}
    days = _results(context, "ProduceHifDayText", image)
    if len(_results(context, "ProduceHifTitleText", image)) != 1:
        return {"scene": "unknown", "reason": "hif_scenario_not_observed"}
    if len(days) != 1:
        return {"scene": "unknown", "reason": "day_not_uniquely_observed"}
    day_match = re.fullmatch(r"(?:H[.．]?I[.．]?F[.．]?本戦まで)?([0-6])日", re.sub(r"\s+", "", days[0].text))
    if not day_match:
        return {"scene": "unknown", "reason": "unknown_day"}
    day = day_from_countdown(int(day_match.group(1)))
    matches = {}
    for scene, node, kind in (
        ("lesson", "ProduceHifLessonText", "lesson"),
        ("training", "ProduceHifTrainingText", "training"),
        ("outing", "ProduceHifOutingText", "outing"),
        ("consult_entry", "ProduceHifConsultText", "consult"),
    ):
        results = _results(context, node, image)
        if results:
            matches[scene] = (results, kind)
    if len(matches) != 1:
        return {"scene": "unknown", "day": day, "reason": "scene_not_uniquely_observed"}
    scene, (results, kind) = next(iter(matches.items()))
    if scene in {"lesson", "training"}:
        # Attribute and SP availability need their own calibrated producer.
        return {"scene": scene, "day": day, "targets": [], "reason": "attribute_and_availability_producer_required"}
    if len(results) != 1:
        return {"scene": scene, "day": day, "targets": [], "reason": "target_not_unique"}
    box = list(results[0].box)
    return {
        "scene": scene, "scenario": "HIF", "day": day, "reason": "candidate_requires_identity_and_stability",
        "targets": [{"key": kind, "kind": kind, "available": True, "box": box}],
    }
