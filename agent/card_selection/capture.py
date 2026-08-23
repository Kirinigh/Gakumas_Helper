"""Card-identity capture adapter backed by the shared local capture store.

The adapter never acquires a frame and never exposes an input method. Callers
must supply both an enabled store and a valid batch identifier before any image
is encoded or any directory/session is created.
"""

from __future__ import annotations

import io
import re
import uuid
import hashlib
from typing import Any, Mapping, Protocol, Sequence
from pathlib import Path
from datetime import UTC, datetime
from dataclasses import dataclass

from .types import CandidateBox, CardPrediction

CAPTURE_CONTRACT = "task100-card-capture-v2"
CAPTURE_PRODUCER_ID = "card_identity_capture_v1"
_BATCH_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_ROLE_PARTITION = {
    "reward_card_runtime": "unassigned",
    "intake_smoke": "intake",
    "atlas_validation": "validation",
}
_SOURCE_TYPES = {"dmm_pc", "synthetic_test"}


def new_capture_batch_id(now: datetime | None = None) -> str:
    """Create one non-secret batch identifier for a single runtime observation."""

    instant = now or datetime.now(UTC)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("capture batch time must include a timezone")
    stamp = instant.astimezone(UTC).strftime("%Y%m%dt%H%M%S%fz")
    return f"task100-{stamp}-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class CaptureEnvironment:
    controller_profile: str
    target_environment: str
    resource_profile: str
    resource_loaded: bool
    resource_hash: str | None = None
    window_class: str | None = None
    process_name: str | None = None
    client_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.controller_profile not in {"PC", "synthetic"}:
            raise ValueError("unsupported controller profile")
        if self.target_environment not in {"DMM_PC", "synthetic_test"}:
            raise ValueError("unsupported target environment")
        if self.resource_profile not in {"DMM", "none", "synthetic"}:
            raise ValueError("unsupported resource profile")
        if self.target_environment == "DMM_PC" and self.controller_profile != "PC":
            raise ValueError("DMM_PC capture requires the PC controller")
        if self.resource_loaded and self.resource_profile != "DMM":
            raise ValueError("only the DMM resource may be marked loaded for a DMM capture")
        if self.target_environment == "DMM_PC" and (
            self.window_class != "UnityWndClass"
            or self.process_name != "gakumas.exe"
            or self.client_size != (720, 1280)
        ):
            raise ValueError("DMM_PC capture requires verified Gakumas window facts")

    @classmethod
    def dmm_runtime(
        cls,
        *,
        resource_hash: str,
        window_class: str,
        process_name: str,
        client_size: tuple[int, int],
    ) -> CaptureEnvironment:
        if not isinstance(resource_hash, str) or not resource_hash.strip():
            raise ValueError("loaded DMM resource hash is required")
        return cls("PC", "DMM_PC", "DMM", True, resource_hash.strip(), window_class, process_name, client_size)

    @classmethod
    def dmm_intake(
        cls,
        *,
        window_class: str,
        process_name: str,
        client_size: tuple[int, int],
    ) -> CaptureEnvironment:
        return cls("PC", "DMM_PC", "none", False, None, window_class, process_name, client_size)

    @classmethod
    def synthetic(cls) -> CaptureEnvironment:
        return cls("synthetic", "synthetic_test", "synthetic", False)


class CaptureStore(Protocol):
    config: Any

    def start_session(self, frame_version: str) -> Any: ...

    def write_image(
        self,
        category: str,
        session: Any,
        data: bytes,
        *,
        metadata: Mapping[str, Any] | None = None,
        capture_kind: str | None = None,
        suffix: str = ".bin",
    ) -> Path: ...


@dataclass(frozen=True)
class EncodedImage:
    data: bytes
    shape: tuple[int, int, int]


def encode_png(image: Any, box: tuple[int, int, int, int] | None = None) -> EncodedImage:
    """Encode one Maa BGR/BGRA frame or ROI as lossless PNG in memory."""

    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("numpy and Pillow are required for local image capture") from error

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError("capture frame must be uint8 HxWx3 or HxWx4")
    if box is not None:
        x, y, width, height = box
        frame_height, frame_width = array.shape[:2]
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
            raise ValueError("capture ROI is empty or outside the frame")
        array = array[y : y + height, x : x + width]

    if array.shape[2] == 3:
        converted = np.ascontiguousarray(array[:, :, ::-1])
        mode = "RGB"
    else:
        converted = np.ascontiguousarray(array[:, :, [2, 1, 0, 3]])
        mode = "RGBA"

    buffer = io.BytesIO()
    Image.fromarray(converted, mode=mode).save(buffer, format="PNG")
    return EncodedImage(buffer.getvalue(), tuple(int(value) for value in converted.shape))


class CardCaptureAdapter:
    """Apply card-identity capture semantics without creating a second writer."""

    def __init__(
        self,
        store: CaptureStore,
        *,
        batch_id: str | None,
        requested_role: str,
        source_type: str = "dmm_pc",
        environment: CaptureEnvironment | None = None,
    ) -> None:
        if batch_id is not None and (not isinstance(batch_id, str) or _BATCH_ID.fullmatch(batch_id) is None):
            raise ValueError("capture_batch_id must match [a-z0-9][a-z0-9._-]{0,63}")
        if requested_role not in _ROLE_PARTITION:
            raise ValueError(f"unsupported capture role: {requested_role}")
        if source_type not in _SOURCE_TYPES:
            raise ValueError(f"unsupported source type: {source_type}")
        self._store = store
        self.batch_id = batch_id
        self.requested_role = requested_role
        self.source_type = source_type
        capture_requested = bool(batch_id) and bool(store.config.enabled)
        if environment is None and source_type == "dmm_pc" and capture_requested:
            raise ValueError("dmm_pc capture requires verified environment facts")
        self.environment = environment or CaptureEnvironment.synthetic()
        if source_type == "dmm_pc" and capture_requested and self.environment.target_environment != "DMM_PC":
            raise ValueError("dmm_pc capture requires a verified DMM_PC environment")
        if requested_role == "reward_card_runtime" and source_type == "dmm_pc" and capture_requested and not (
            self.environment.resource_loaded and self.environment.resource_profile == "DMM"
        ):
            raise ValueError("reward-card runtime capture requires a loaded DMM resource")
        if requested_role in {"intake_smoke", "atlas_validation"} and self.environment.resource_loaded:
            raise ValueError("read-only intake must not claim a loaded recognition resource")
        self._session: Any | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.batch_id) and bool(self._store.config.enabled)

    @property
    def normal_capture_enabled(self) -> bool:
        return self.enabled and self._store.config.mode in {"roi", "screenshot"}

    @property
    def failure_capture_enabled(self) -> bool:
        return self.enabled and self._store.config.mode in {"failure", "roi", "screenshot"}

    @property
    def session_id(self) -> str | None:
        return None if self._session is None else str(self._session.session_id)

    def capture_normal_frame(
        self,
        image: Any,
        *,
        frame_id: str,
        capture_index: int,
        candidates: Sequence[CandidateBox],
        predictions: Sequence[CardPrediction],
    ) -> tuple[Path, ...]:
        """Persist an ordinary reward-card frame according to the selected mode."""

        if not self.normal_capture_enabled:
            return ()
        prediction_by_slot = {item.slot: item for item in predictions}
        payloads: list[tuple[str, EncodedImage, dict[str, Any]]] = []
        group_id = f"{frame_id}-normal"
        group_expected = len(candidates) + (1 if self._store.config.mode == "screenshot" else 0)
        if self._store.config.mode == "screenshot":
            payloads.append(
                (
                    "screenshot",
                    encode_png(image),
                    self._base_metadata(
                        image,
                        frame_id=frame_id,
                        capture_index=capture_index,
                        image_role="reward_card_screenshot",
                        candidate_count=len(candidates),
                        expected_candidate_count=3,
                        outcome="OBSERVE",
                        reason="candidate_layout_valid",
                        group_id=group_id,
                        group_expected=group_expected,
                        group_index=0,
                    ),
                )
            )
        for candidate in candidates:
            prediction = prediction_by_slot.get(candidate.slot)
            metadata = self._base_metadata(
                image,
                frame_id=frame_id,
                capture_index=capture_index,
                image_role="candidate_roi",
                candidate_count=len(candidates),
                expected_candidate_count=3,
                outcome="OBSERVE",
                reason="candidate_layout_valid",
                group_id=group_id,
                group_expected=group_expected,
                group_index=candidate.slot + (1 if self._store.config.mode == "screenshot" else 0),
            )
            metadata.update(
                {
                    "slot": candidate.slot,
                    "box": list(candidate.box),
                    "detector_label": candidate.detector_label,
                    "detector_score": candidate.detector_score,
                    "prediction": None
                    if prediction is None
                    else {
                        "card_id": prediction.card_id,
                        "predicted_card_id": prediction.predicted_card_id,
                        "class_name": prediction.class_name,
                        "confidence": prediction.confidence,
                        "margin": prediction.margin,
                        "accepted": prediction.accepted,
                        "reason": prediction.reason,
                        "model_sha256": prediction.model_sha256,
                        "classes_sha256": prediction.classes_sha256,
                        "recognizer": prediction.recognizer,
                        "gallery_sha256": prediction.gallery_sha256,
                        "top_k_card_ids": [list(group) for group in prediction.top_k_card_ids],
                        "top_k_scores": list(prediction.top_k_scores),
                    },
                }
            )
            payloads.append(("roi", encode_png(image, candidate.box), metadata))
        return self._write_normal(payloads)

    def capture_failure_frame(
        self,
        image: Any,
        *,
        frame_id: str,
        capture_index: int,
        reason: str,
        candidate_count: int | None,
    ) -> tuple[Path, ...]:
        """Persist exactly one full failure scene; callers must not retry a failed write."""

        if not self.failure_capture_enabled:
            return ()
        encoded = encode_png(image)
        metadata = self._base_metadata(
            image,
            frame_id=frame_id,
            capture_index=capture_index,
            image_role="failure_scene",
            candidate_count=candidate_count,
            expected_candidate_count=3,
            outcome="STOP",
            reason=reason,
        )
        return (self._write("failure", "failure_scene", encoded, metadata),)

    def capture_intake_frame(
        self,
        image: Any,
        *,
        frame_id: str,
        reason: str,
        capture_index: int = 0,
        expected_candidate_count: int = 0,
        outcome: str = "STOP",
        image_role: str = "intake_screenshot",
        executed: bool = False,
    ) -> tuple[Path, ...]:
        """Persist one input-disabled read-only DMM screenshot."""

        if self.requested_role not in {"intake_smoke", "atlas_validation"}:
            raise ValueError("intake capture requires a read-only intake role")
        if outcome not in {"STOP", "OBSERVE"}:
            raise ValueError("read-only intake outcome must be STOP or OBSERVE")
        if not self.enabled or self._store.config.mode != "screenshot":
            return ()
        encoded = encode_png(image)
        metadata = self._base_metadata(
            image,
            frame_id=frame_id,
            capture_index=capture_index,
            image_role=image_role,
            candidate_count=None,
            expected_candidate_count=expected_candidate_count,
            outcome=outcome,
            reason=reason,
            executed=executed,
        )
        return (self._write("normal", "screenshot", encoded, metadata),)

    def _write_normal(self, payloads: Sequence[tuple[str, EncodedImage, dict[str, Any]]]) -> tuple[Path, ...]:
        paths: list[Path] = []
        for capture_kind, encoded, metadata in payloads:
            paths.append(self._write("normal", capture_kind, encoded, metadata))
        return tuple(paths)

    def _write(self, category: str, capture_kind: str, encoded: EncodedImage, metadata: dict[str, Any]) -> Path:
        if self._session is None:
            self._session = self._store.start_session(CAPTURE_CONTRACT)
        metadata = dict(metadata)
        metadata["image_shape"] = list(encoded.shape)
        metadata["content_sha256"] = hashlib.sha256(encoded.data).hexdigest().upper()
        return self._store.write_image(
            category,
            self._session,
            encoded.data,
            metadata=metadata,
            capture_kind=capture_kind,
            suffix=".png",
        )

    def _base_metadata(
        self,
        image: Any,
        *,
        frame_id: str,
        capture_index: int,
        image_role: str,
        candidate_count: int | None,
        expected_candidate_count: int,
        outcome: str,
        reason: str,
        group_id: str | None = None,
        group_expected: int = 1,
        group_index: int = 0,
        executed: bool = False,
    ) -> dict[str, Any]:
        shape = getattr(image, "shape", ())
        return {
            "task_id": CAPTURE_PRODUCER_ID,
            "capture_contract": CAPTURE_CONTRACT,
            "source_type": self.source_type,
            "controller_profile": self.environment.controller_profile,
            "target_environment": self.environment.target_environment,
            "resource_profile": self.environment.resource_profile,
            "resource_loaded": self.environment.resource_loaded,
            "resource_hash": self.environment.resource_hash,
            "target_window_class": self.environment.window_class,
            "target_process_name": self.environment.process_name,
            "target_client_size": None if self.environment.client_size is None else list(self.environment.client_size),
            "capture_batch_id": self.batch_id,
            "requested_role": self.requested_role,
            "annotation_state": "pending_manual",
            "ground_truth": None,
            "partition": _ROLE_PARTITION[self.requested_role],
            "frame_id": frame_id,
            "capture_index": capture_index,
            "source_frame_shape": [int(value) for value in shape],
            "image_role": image_role,
            "candidate_count": candidate_count,
            "expected_candidate_count": expected_candidate_count,
            "outcome": outcome,
            "reason": reason,
            "capture_group_id": group_id or f"{frame_id}-{image_role}",
            "capture_group_expected": group_expected,
            "capture_group_index": group_index,
            "read_only": True,
            "executed": executed,
            "click_intent": False,
        }
