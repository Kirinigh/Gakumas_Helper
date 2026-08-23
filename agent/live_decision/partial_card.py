from __future__ import annotations

import statistics
from typing import Any, Iterable
from dataclasses import replace, dataclass

try:
    from agent.card_selection.embedding import focus_card_art
except ModuleNotFoundError:  # Maa runtime exposes ``agent/`` as the import root.
    from card_selection.embedding import focus_card_art

MIN_LIVE_HAND_SIZE = 1
MAX_LIVE_HAND_SIZE = 5
MIN_SUPPORTED_VISIBLE_RATIO = 0.55


class LiveCardLayoutError(ValueError):
    """Raised when Live card geometry cannot be reconstructed safely."""


@dataclass(frozen=True)
class LivePartialCardView:
    """One card-art crop with its occluded width restored as a white mask."""

    slot: int
    candidate_box: tuple[int, int, int, int]
    visible_box: tuple[int, int, int, int]
    canvas_size: tuple[int, int]
    visible_ratio: float
    view_mode: str
    layout_version: str


def normalise_live_candidates(candidates: Iterable[Any]) -> tuple[Any, ...]:
    """Sort and deduplicate 1..5 Live candidates without silently truncating them."""

    ordered = sorted(candidates, key=lambda candidate: (candidate.box[0], candidate.box[1]))
    deduplicated: list[Any] = []
    for candidate in ordered:
        x, _y, width, _height = candidate.box
        if width <= 0:
            continue
        centre = x + width / 2.0
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(deduplicated)
                if abs(centre - (existing.box[0] + existing.box[2] / 2.0))
                <= 0.18 * min(width, existing.box[2])
            ),
            None,
        )
        if duplicate_index is None:
            deduplicated.append(candidate)
        elif candidate.detector_score > deduplicated[duplicate_index].detector_score:
            deduplicated[duplicate_index] = candidate
    if not MIN_LIVE_HAND_SIZE <= len(deduplicated) <= MAX_LIVE_HAND_SIZE:
        return ()
    deduplicated.sort(key=lambda candidate: (candidate.box[0], candidate.box[1]))
    return tuple(replace(candidate, slot=slot) for slot, candidate in enumerate(deduplicated))


def _view_mode(visible_ratio: float) -> str:
    if visible_ratio >= 0.90:
        return "full"
    return min((0.55, 0.65, 0.75), key=lambda value: abs(value - visible_ratio)).__format__(
        ".2f"
    )


def _validate_box(
    box: tuple[int, int, int, int], frame_width: int, frame_height: int
) -> None:
    x, y, width, height = box
    if (
        x < 0
        or y < 0
        or width < 8
        or height < 8
        or x + width > frame_width
        or y + height > frame_height
    ):
        raise LiveCardLayoutError("Live partial-card ROI is outside the frame")


def build_live_card_views(
    frame_shape: tuple[int, ...],
    candidates: Iterable[Any],
    *,
    legacy_art_boxes: Iterable[tuple[int, int, int, int]] | None = None,
) -> tuple[LivePartialCardView, ...]:
    """Reconstruct canonical card-art views for an unoccluded or overlapping hand."""

    if len(frame_shape) < 2:
        raise LiveCardLayoutError("Live frame shape is invalid")
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    cards = normalise_live_candidates(candidates)
    if not cards:
        raise LiveCardLayoutError("Live hand must contain one to five unique cards")

    legacy = tuple(legacy_art_boxes or ())
    if legacy:
        if len(legacy) != len(cards):
            raise LiveCardLayoutError("legacy Live art boxes do not match the hand")
        views = []
        for card, box in zip(cards, legacy, strict=True):
            _validate_box(box, frame_width, frame_height)
            views.append(
                LivePartialCardView(
                    slot=card.slot,
                    candidate_box=card.box,
                    visible_box=box,
                    canvas_size=(box[2], box[3]),
                    visible_ratio=1.0,
                    view_mode="full",
                    layout_version="dmm-live-legacy-art-v1",
                )
            )
        return tuple(views)

    if len(cards) <= 3:
        views = []
        for card in cards:
            x, y, width, height = card.box
            art_width = round(width * 0.95)
            art_height = min(art_width, round(height * 0.74))
            box = (
                x + round(width * 0.025),
                y + round(height * 0.06),
                art_width,
                art_height,
            )
            _validate_box(box, frame_width, frame_height)
            views.append(
                LivePartialCardView(
                    slot=card.slot,
                    candidate_box=card.box,
                    visible_box=box,
                    canvas_size=(art_width, art_height),
                    visible_ratio=1.0,
                    view_mode="full",
                    layout_version="dmm-live-unoccluded-v1",
                )
            )
        return tuple(views)

    anchors = [card.box[0] for card in cards]
    spacings = [right - left for left, right in zip(anchors, anchors[1:])]
    spacing = float(statistics.median(spacings))
    if spacing < 24 or any(abs(value - spacing) > max(6.0, spacing * 0.28) for value in spacings):
        raise LiveCardLayoutError("Live overlapping hand spacing is inconsistent")

    maximum_detector_width = max(card.box[2] for card in cards)
    row_top = min(card.box[1] for card in cards)
    if len(cards) == 4:
        full_width = min(
            max(maximum_detector_width, round(spacing * 1.05)),
            round(spacing * 1.35),
        )
        art_top = row_top + round(full_width * 0.13)
        art_height = full_width
        horizontal_inset = round(full_width * 0.05)
        layout_version = "dmm-live-overlap-four-v1"
    else:
        full_width = round(statistics.median((maximum_detector_width, spacing * 1.38)))
        full_width = min(max(full_width, round(spacing * 1.20)), round(spacing * 1.60))
        art_top = row_top + round(spacing * 0.15)
        art_height = round(spacing * 1.40)
        horizontal_inset = round(spacing * 0.075)
        layout_version = "dmm-live-overlap-v1"
    neighbour_gap = max(2, round(spacing * 0.02))

    views = []
    for index, card in enumerate(cards):
        left = card.box[0] + horizontal_inset
        right = (
            anchors[index + 1] - neighbour_gap
            if index + 1 < len(cards)
            else min(card.box[0] + full_width, frame_width)
        )
        visible_width = right - left
        visible_ratio = visible_width / full_width
        if visible_ratio < MIN_SUPPORTED_VISIBLE_RATIO or visible_ratio > 1.05:
            raise LiveCardLayoutError(
                f"Live card slot {card.slot} has unsupported visible ratio {visible_ratio:.3f}"
            )
        box = (left, art_top, visible_width, art_height)
        _validate_box(box, frame_width, frame_height)
        views.append(
            LivePartialCardView(
                slot=card.slot,
                candidate_box=card.box,
                visible_box=box,
                canvas_size=(full_width, art_height),
                visible_ratio=min(visible_ratio, 1.0),
                view_mode=_view_mode(min(visible_ratio, 1.0)),
                layout_version=layout_version,
            )
        )
    return tuple(views)


def preprocess_live_card_view(
    image: Any,
    view: LivePartialCardView,
    *,
    source_color_order: str,
) -> Any:
    """Return one canonical 64x64 RGB tensor without stretching visible fragments."""

    import numpy as np
    from PIL import Image

    if source_color_order not in {"RGB", "BGR"}:
        raise ValueError("source_color_order must be RGB or BGR")
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] not in {3, 4}:
        raise ValueError("Live frame must be HxWx3 or HxWx4")
    x, y, width, height = view.visible_box
    _validate_box(view.visible_box, frame.shape[1], frame.shape[0])
    crop = frame[y : y + height, x : x + width, :3]
    if source_color_order == "BGR":
        crop = crop[:, :, ::-1]
    canvas = Image.new("RGB", view.canvas_size, "white")
    canvas.paste(Image.fromarray(np.ascontiguousarray(crop, dtype=np.uint8), mode="RGB"), (0, 0))
    focused = focus_card_art(canvas).resize((64, 64), Image.Resampling.BILINEAR)
    rgb = np.asarray(focused, dtype=np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None, ...]


def classify_live_card_views(
    recognizer: Any,
    image: Any,
    candidates: Iterable[Any],
    *,
    frame_id: str,
    plan: str,
    acceptance_threshold: float | None,
    min_margin: float | None,
    legacy_art_boxes: Iterable[tuple[int, int, int, int]] | None = None,
) -> tuple[Any, ...]:
    """Return visual-family predictions; Live upgrade resolution remains external."""

    cards = normalise_live_candidates(candidates)
    views = build_live_card_views(
        tuple(image.shape), cards, legacy_art_boxes=legacy_art_boxes
    )
    predictions = []
    for card, view in zip(cards, views, strict=True):
        tensor = preprocess_live_card_view(
            image,
            view,
            source_color_order=recognizer.embedder.source_color_order,
        )
        prediction = recognizer.classify_embedding(
            recognizer.embedder.embed_tensor(tensor),
            card,
            frame_id=frame_id,
            plan=plan,
            acceptance_threshold=acceptance_threshold,
            min_margin=min_margin,
            resolve_upgrade_state=False,
        )
        predictions.append(
            replace(
                prediction,
                recognizer="card_art_embedding_top5_live_partial_v1",
                view_mode=view.view_mode,
                visible_ratio=view.visible_ratio,
            )
        )
    return tuple(predictions)
