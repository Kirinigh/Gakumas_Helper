from __future__ import annotations

import hashlib
from typing import Any, Sequence
from pathlib import Path

from .types import CandidateBox


class ModelIntegrityError(RuntimeError):
    """Raised before inference when a pinned asset differs from its manifest."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def frame_identifier(image: Any, capture_index: int = 0) -> str:
    """Return a deterministic, non-secret frame ID without writing the frame."""

    try:
        payload = memoryview(image).tobytes()
        shape = str(tuple(image.shape)).encode("ascii")
    except (TypeError, AttributeError):
        payload = bytes(image)
        shape = b"unknown"
    digest = hashlib.sha256(shape + payload).hexdigest()[:16]
    return f"frame-{capture_index:02d}-{digest}"


def isolate_card_candidates(
    results: Sequence[Any],
    *,
    target_count: int | None = None,
) -> tuple[CandidateBox, ...]:
    """Enumerate selectable cards; raw labels are audit-only and ``useless`` is excluded."""

    candidates: list[tuple[tuple[int, int, int, int], float, str]] = []
    for result in results:
        label = result.get("label") if isinstance(result, dict) else getattr(result, "label", None)
        if label not in {"cards", "suggestions"}:
            continue
        raw_box = result.get("box") if isinstance(result, dict) else getattr(result, "box", None)
        raw_score = result.get("score", 0.0) if isinstance(result, dict) else getattr(result, "score", 0.0)
        if raw_box is None or len(raw_box) != 4:
            continue
        box = tuple(int(round(value)) for value in raw_box)
        candidates.append((box, float(raw_score), str(label)))
    candidates.sort(key=lambda item: (item[0][0], item[0][1]))
    if target_count is not None and len(candidates) > target_count:
        deduplicated: list[tuple[tuple[int, int, int, int], float, str]] = []
        for candidate in candidates:
            box, score, _label = candidate
            x, _y, width, _height = box
            center = x + width / 2
            match_index = next(
                (
                    index
                    for index, (existing_box, _existing_score, _existing_label) in enumerate(deduplicated)
                    if abs(center - (existing_box[0] + existing_box[2] / 2)) <= 0.55 * max(width, existing_box[2])
                ),
                None,
            )
            if match_index is None:
                deduplicated.append(candidate)
            elif score > deduplicated[match_index][1]:
                deduplicated[match_index] = candidate
        if len(deduplicated) == target_count:
            candidates = sorted(deduplicated, key=lambda item: (item[0][0], item[0][1]))
    return tuple(
        CandidateBox(slot=index, box=box, detector_label=label, detector_score=score) for index, (box, score, label) in enumerate(candidates)
    )
