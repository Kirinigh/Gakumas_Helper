"""Pure evidence checks for a frozen P-item row and its restored page."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from collections.abc import Sequence

from p_item_recognition import PItemReferenceError, measure_p_item_content_generation

from .reader import ArenaReaderError
from ._reader_visual import _p_item_border_only_change, _p_item_source_proof_boxes


@dataclass(frozen=True)
class PItemSourceEvidence:
    matches: bool
    errors: tuple[float, ...]
    border_animation_accepted: bool = False


def p_item_source_generation_evidence(
    source_images: Sequence[Any],
    boxes: Sequence[tuple[int, int, int, int]],
    *,
    threshold: float,
) -> PItemSourceEvidence:
    """Measure the original frozen window, without admitting border changes."""

    if len(source_images) < 2:
        return PItemSourceEvidence(True, ())
    try:
        proof_boxes = _p_item_source_proof_boxes(source_images[0], boxes)
        errors = measure_p_item_content_generation(source_images, proof_boxes)
    except (PItemReferenceError, TypeError, ValueError) as error:
        raise ArenaReaderError(
            "p_item_source_restore_evidence_invalid",
            "P-item frozen source-page evidence could not be measured",
        ) from error
    return PItemSourceEvidence(not any(value > threshold for value in errors), errors)


def p_item_source_frame_evidence(
    source_images: Sequence[Any],
    image: Any,
    boxes: Sequence[tuple[int, int, int, int]],
    *,
    member_anchors_visible: bool,
    threshold: float,
) -> PItemSourceEvidence:
    """Require one complete source match with anchors from this current frame."""

    if not boxes:
        raise ArenaReaderError(
            "p_item_source_restore_evidence_missing",
            "P-item source-page evidence requires the accepted source-row boxes",
        )
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
    proof_boxes = _p_item_source_proof_boxes(source_images[0], boxes)
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
            if member_anchors_visible and all(value <= threshold for value in errors):
                return PItemSourceEvidence(True, errors)
            if (
                member_anchors_visible
                and all(value <= threshold for value in errors[len(boxes):])
                and _p_item_border_only_change(source_image, image, boxes, threshold)
            ):
                return PItemSourceEvidence(True, errors, border_animation_accepted=True)
    except (PItemReferenceError, TypeError, ValueError) as error:
        raise ArenaReaderError(
            "p_item_source_restore_evidence_invalid",
            "P-item source-row restore evidence could not be measured",
        ) from error
    return PItemSourceEvidence(False, best_errors)
