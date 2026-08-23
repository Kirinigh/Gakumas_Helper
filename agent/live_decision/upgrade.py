from __future__ import annotations

from typing import Any
from dataclasses import asdict, dataclass

BASE_FRAME_SIZE = (720, 1280)
LIVE_CARD_NAME_BOXES = (
    (40, 1090, 210, 38),
    (250, 1090, 225, 38),
    (465, 1090, 235, 38),
)


@dataclass(frozen=True)
class LiveUpgradeObservation:
    """One colour-independent observation of a Live card-name suffix."""

    upgraded: bool
    marker_count: int
    confidence: float
    box: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveUpgradeConsensus:
    """Fail-closed normal/+ result from three nearby Live frames."""

    accepted: bool
    upgraded: bool | None
    plus_votes: int
    normal_votes: int
    reason: str
    observations: tuple[LiveUpgradeObservation, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scaled_name_crop(
    image: Any,
    slot: int,
    box: tuple[int, int, int, int] | None = None,
) -> tuple[Any, tuple[int, int, int, int]]:
    import numpy as np

    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("Live frame must be a colour image")
    frame_height, frame_width = frame.shape[:2]
    if box is None:
        if slot < 0 or slot >= len(LIVE_CARD_NAME_BOXES):
            raise ValueError(f"unsupported Live card slot: {slot}")
        base_width, base_height = BASE_FRAME_SIZE
        left, top, width, height = LIVE_CARD_NAME_BOXES[slot]
        scaled = (
            round(left * frame_width / base_width),
            round(top * frame_height / base_height),
            round(width * frame_width / base_width),
            round(height * frame_height / base_height),
        )
    else:
        scaled = tuple(int(value) for value in box)
    x, y, scaled_width, scaled_height = scaled
    if x < 0 or y < 0 or x + scaled_width > frame_width or y + scaled_height > frame_height:
        raise ValueError("Live card-name ROI is outside the frame")
    crop = frame[y : y + scaled_height, x : x + scaled_width, :3]
    if box is None and (scaled_width, scaled_height) != (width, height):
        rows = np.linspace(0, scaled_height - 1, height).round().astype(int)
        columns = np.linspace(0, scaled_width - 1, width).round().astype(int)
        crop = crop[rows][:, columns]
    return crop, scaled


def _connected_components(mask: Any) -> list[tuple[int, int, int, int, int, Any]]:
    import numpy as np

    foreground = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(foreground, dtype=bool)
    frame_height, frame_width = foreground.shape
    components = []
    for seed_y, seed_x in np.argwhere(foreground):
        y, x = int(seed_y), int(seed_x)
        if visited[y, x]:
            continue
        visited[y, x] = True
        stack = [(y, x)]
        pixels: list[tuple[int, int]] = []
        while stack:
            current_y, current_x = stack.pop()
            pixels.append((current_y, current_x))
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    if offset_y == 0 and offset_x == 0:
                        continue
                    neighbour_y = current_y + offset_y
                    neighbour_x = current_x + offset_x
                    if not (0 <= neighbour_y < frame_height and 0 <= neighbour_x < frame_width):
                        continue
                    if foreground[neighbour_y, neighbour_x] and not visited[neighbour_y, neighbour_x]:
                        visited[neighbour_y, neighbour_x] = True
                        stack.append((neighbour_y, neighbour_x))
        rows = np.fromiter((item[0] for item in pixels), dtype=int)
        columns = np.fromiter((item[1] for item in pixels), dtype=int)
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(columns.min()), int(columns.max())
        component = np.zeros((bottom - top + 1, right - left + 1), dtype=bool)
        component[rows - top, columns - left] = True
        components.append(
            (left, top, right - left + 1, bottom - top + 1, len(pixels), component)
        )
    return components


def detect_live_upgrade_marker(
    image: Any,
    slot: int,
    *,
    box: tuple[int, int, int, int] | None = None,
    expanded: bool = False,
) -> LiveUpgradeObservation:
    """Detect one or more suffix ``+`` glyphs without depending on their colour.

    The Live title panel renders both the permanent black marker and the
    temporary Support-effect blue marker as the same compact cross. The mask
    therefore uses channel-independent darkness and checks cross geometry only.
    """

    import numpy as np

    crop, scaled_box = _scaled_name_crop(image, slot, box)
    # ``min`` is invariant to RGB/BGR ordering and keeps both black and blue ink.
    foreground = np.min(crop.astype(np.uint8), axis=2) < (190 if expanded else 180)
    components = _connected_components(foreground)
    boundary_margin = 4 if expanded else 8

    text_components = [
        item
        for item in components
        if item[4] >= 3
        and 3 <= item[3] <= 25
        and item[1] >= 10
        and item[0] + item[2] <= crop.shape[1] - boundary_margin
    ]
    rightmost_ink = max((left + width for left, _, width, _, _, _ in text_components), default=0)
    scores: list[float] = []
    for left, _top, width, height, area, component in components:
        if left + width > crop.shape[1] - boundary_margin or left < 45:
            continue
        row_counts = component.sum(axis=1)
        column_counts = component.sum(axis=0)
        row_index = int(np.argmax(row_counts))
        column_index = int(np.argmax(column_counts))
        row_score = float(row_counts[row_index] / width)
        column_score = float(column_counts[column_index] / height)
        compact_cross = (
            7 <= width <= 13
            and 7 <= height <= 13
            and 15 <= area <= 90
            and 0.65 <= width / height <= 1.45
            and row_score >= 0.8
            and column_score >= 0.8
            and abs(row_index - (height - 1) / 2) <= 2.5
            and abs(column_index - (width - 1) / 2) <= 2.5
        )
        expanded_cross = (
            expanded
            and 14 <= width <= 19
            and 12 <= height <= 19
            and 70 <= area <= 150
            and 0.75 <= width / height <= 1.35
            and row_score >= 0.9
            and column_score >= 0.7
            and row_index <= 3
            and abs(column_index - (width - 1) / 2) <= 3.5
        )
        if not (compact_cross or expanded_cross):
            continue
        if left + width < rightmost_ink - 2:
            continue
        scores.append(min(row_score, column_score))

    return LiveUpgradeObservation(
        upgraded=bool(scores),
        marker_count=len(scores),
        confidence=max(scores, default=1.0),
        box=scaled_box,
    )


def resolve_live_upgrade_consensus(
    images: Any,
    slot: int,
    *,
    box: tuple[int, int, int, int] | None = None,
    expanded: bool = False,
) -> LiveUpgradeConsensus:
    """Collapse ``+``/``++``/``+++`` to one enhanced state over three frames."""

    frames = tuple(images)
    if len(frames) != 3:
        raise ValueError("Live upgrade consensus requires exactly three frames")
    observations = tuple(
        detect_live_upgrade_marker(image, slot, box=box, expanded=expanded)
        for image in frames
    )
    plus_votes = sum(observation.upgraded for observation in observations)
    normal_votes = len(observations) - plus_votes
    if plus_votes >= 2:
        return LiveUpgradeConsensus(
            True,
            True,
            plus_votes,
            normal_votes,
            "plus_majority",
            observations,
        )
    if plus_votes == 0:
        return LiveUpgradeConsensus(
            True,
            False,
            0,
            normal_votes,
            "normal_unanimous",
            observations,
        )
    return LiveUpgradeConsensus(
        False,
        None,
        plus_votes,
        normal_votes,
        "upgrade_frames_disagree",
        observations,
    )
