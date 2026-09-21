"""Shared immutable OCR evidence types; no controller dependency."""
from typing import Any, NamedTuple

_PItemPanelRow = tuple[
    str,
    tuple[float, float, float, float],
    tuple[tuple[str, tuple[float, float, float, float]], ...],
]


_NormalizedPItemRow = tuple[
    str,
    str,
    tuple[float, float, float, float],
    tuple[tuple[str, str, tuple[float, float, float, float]], ...],
]


class _PItemOcrObservation(NamedTuple):
    """One OCR call, with separate title and size-excluded source evidence."""

    full_text: str
    panel_text: str
    anchors_visible: bool
    panel_rows: tuple[_PItemPanelRow, ...]
    background_rows: tuple[_PItemPanelRow, ...] = ()
    all_items: tuple[Any, ...] = ()
