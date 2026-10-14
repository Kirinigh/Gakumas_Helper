"""Keep the packaged arena detector independent from upstream card detection."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

PIPELINE_PATH = "resource/base/pipeline/ArenaCards.json"
MODEL_PATH = "resource/base/model/detect/arena_cards_v1.onnx"
MODEL_SHA256 = "44f0af9247df55b423cec41999baa8a655e67e86f590f28f0e852c1fef62f0f7"
UPSTREAM_PIPELINE_PATH = "resource/base/pipeline/ProduceUtils.json"
CRITICAL_PATHS = (PIPELINE_PATH, MODEL_PATH)
NODE_NAME = "ArenaRecognitionCards"

_EXPECTED_NODE = {
    "recognition": {
        "type": "NeuralNetworkDetect",
        "param": {
            "model": "arena_cards_v1.onnx",
            "expected": [0, 1, 2],
            "labels": ["cards", "suggestions", "useless"],
            "threshold": [0.3, 0.3, 0.3],
        },
    },
    "action": {"type": "DoNothing"},
}


class ArenaDetectorContractError(ValueError):
    """A candidate lost the independent arena detector contract."""


def _candidate_file(candidate: Path, relative: str) -> Path:
    path = (candidate / relative).resolve()
    if not path.is_relative_to(candidate):
        raise ArenaDetectorContractError(f"candidate arena detector path escapes root: {relative}")
    if not path.is_file():
        raise ArenaDetectorContractError(f"candidate arena detector file is missing: {relative}")
    return path


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArenaDetectorContractError(f"candidate arena detector pipeline has duplicate key: {key}")
        result[key] = value
    return result


def _load_pipeline(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (OSError, UnicodeError, ValueError) as error:
        raise ArenaDetectorContractError(f"candidate arena detector pipeline is invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise ArenaDetectorContractError(f"candidate arena detector pipeline must be an object: {path.name}")
    return value


def validate_candidate_arena_detector(candidate: Path) -> None:
    """Validate installed resources; file hashes stay in the existing inventory."""
    candidate = candidate.resolve()
    pipeline_path = _candidate_file(candidate, PIPELINE_PATH)
    model_path = _candidate_file(candidate, MODEL_PATH)
    try:
        if model_path.stat().st_size == 0:
            raise ArenaDetectorContractError("candidate arena detector model is empty")
        digest = hashlib.sha256()
        with model_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ArenaDetectorContractError("candidate arena detector model is unavailable") from error
    if digest.hexdigest().casefold() != MODEL_SHA256.casefold():
        raise ArenaDetectorContractError("candidate arena detector v1 model SHA-256 does not match its approved source")

    pipeline = _load_pipeline(pipeline_path)
    node = pipeline.get(NODE_NAME)
    if isinstance(node, dict) and node.get("action") == {"type": "DoNothing", "param": {}}:
        node = {**node, "action": {"type": "DoNothing"}}
    if set(pipeline) != {NODE_NAME} or node != _EXPECTED_NODE:
        raise ArenaDetectorContractError("candidate arena detector node contract is invalid")
    # JSON booleans compare equal to integer class IDs in Python.
    expected = node["recognition"]["param"]["expected"]
    if any(type(class_id) is not int for class_id in expected):
        raise ArenaDetectorContractError("candidate arena detector class IDs must be integers")

    upstream = _load_pipeline(_candidate_file(candidate, UPSTREAM_PIPELINE_PATH))
    produce = upstream.get("ProduceRecognitionCards")
    recognition = produce.get("recognition") if isinstance(produce, dict) else None
    param = recognition.get("param") if isinstance(recognition, dict) else None
    if not isinstance(param, dict) or param.get("model") != "cards.onnx":
        raise ArenaDetectorContractError("candidate arena detector must preserve the upstream ProduceRecognitionCards model binding")

    resource_root = candidate / "resource"
    for other in resource_root.rglob("*.json"):
        if "pipeline" not in other.relative_to(resource_root).parts[:-1] or other == candidate / PIPELINE_PATH:
            continue
        other = _candidate_file(candidate, other.relative_to(candidate).as_posix())
        if NODE_NAME in _load_pipeline(other):
            raise ArenaDetectorContractError("candidate arena detector node is overridden by another pipeline")
