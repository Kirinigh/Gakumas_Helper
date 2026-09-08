"""Calibrated, derived full P-item references; this is not an official UI renderer."""

from __future__ import annotations

import math
from typing import Any
from collections.abc import Mapping, Sequence

import cv2
import numpy as np

TOOL_CONTRACT = "task095-calibrated-derived-full-reference-v1"


class CalibratedPItemReferenceError(ValueError):
    """A full-reference image or its catalog pairing is invalid."""


def _validate_image(image: object, channels: int, label: str) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != channels
        or min(image.shape[:2]) < 1
    ):
        raise CalibratedPItemReferenceError(f"Expected nonempty {label} uint8 image")


def compose_enhanced_reference(base_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
    """Replay the frozen full-reference geometry and premultiplied-alpha resampling.

    The calibration fixes width, left and top at 28/58, 34/58 and -4/58 of
    the base width. It does not identify an item or read enhancement in a frame.
    Both inputs remain unchanged, and only the canvas intersection is composed.
    """
    _validate_image(base_bgr, 3, "opaque BGR base")
    _validate_image(overlay_bgra, 4, "straight-alpha BGRA overlay")
    height, width = base_bgr.shape[:2]
    target_width = math.floor(width * 28 / 58 + 0.5)
    target_height = math.floor(target_width * overlay_bgra.shape[0] / overlay_bgra.shape[1] + 0.5)
    left = math.floor(width * 34 / 58 + 0.5)
    top = math.floor(-width * 4 / 58 + 0.5)
    if min(target_width, target_height) < 1:
        raise CalibratedPItemReferenceError("Invalid target size")
    alpha = overlay_bgra[:, :, 3].astype(np.float64) / 255
    premultiplied = overlay_bgra[:, :, :3].astype(np.float64) * alpha[:, :, None]
    alpha = cv2.resize(alpha, (target_width, target_height), interpolation=cv2.INTER_AREA)
    colour = cv2.resize(premultiplied, (target_width, target_height), interpolation=cv2.INTER_AREA)
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + target_width), min(height, top + target_height)
    result = base_bgr.copy()
    if x1 <= x0 or y1 <= y0:
        raise CalibratedPItemReferenceError("Overlay does not intersect canvas")
    a = alpha[y0 - top : y1 - top, x0 - left : x1 - left, None]
    c = colour[y0 - top : y1 - top, x0 - left : x1 - left]
    result[y0:y1, x0:x1] = np.clip(
        np.floor(c + result[y0:y1, x0:x1].astype(np.float64) * (1 - a) + 0.5), 0, 255
    ).astype(np.uint8)
    return result


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def pair_upgraded_p_items(
    catalog_rows: Sequence[Mapping[str, Any]], gallery_ids: Sequence[int]
) -> dict[int, int]:
    """Map each enhanced gallery ID to its unique normal ID in the same gallery.

    Pair identity comes only from (sourceType, pIdolId). Catalog titles, ID
    adjacency and image contents do not participate. Catalog rows outside the
    gallery still count when deciding whether a pair is unique.
    """
    if not isinstance(catalog_rows, Sequence) or isinstance(catalog_rows, (str, bytes)):
        raise CalibratedPItemReferenceError("Catalog rows must be a sequence")
    if not isinstance(gallery_ids, Sequence) or isinstance(gallery_ids, (str, bytes)) or not gallery_ids:
        raise CalibratedPItemReferenceError("Gallery IDs must be a nonempty sequence")
    if not all(_positive_integer(identity) for identity in gallery_ids) or len(set(gallery_ids)) != len(gallery_ids):
        raise CalibratedPItemReferenceError("Gallery IDs must be unique positive integers")
    by_id: dict[int, Mapping[str, Any]] = {}
    groups: dict[tuple[str, int], dict[bool, list[int]]] = {}
    for row in catalog_rows:
        if not isinstance(row, Mapping) or not _positive_integer(row.get("id")):
            raise CalibratedPItemReferenceError("Catalog identity is invalid")
        identity = row["id"]
        if identity in by_id:
            raise CalibratedPItemReferenceError("Catalog identities are duplicated")
        if not isinstance(row.get("upgraded"), bool) or not isinstance(row.get("sourceType"), str):
            raise CalibratedPItemReferenceError("Catalog enhancement or source type is invalid")
        by_id[identity] = row
        if row["sourceType"] == "pIdol":
            if not _positive_integer(row.get("pIdolId")):
                raise CalibratedPItemReferenceError("Catalog pIdolId is invalid")
            key = (row["sourceType"], row["pIdolId"])
            groups.setdefault(key, {False: [], True: []})[row["upgraded"]].append(identity)
    gallery = set(gallery_ids)
    if not gallery.issubset(by_id):
        raise CalibratedPItemReferenceError("Gallery IDs are missing from the catalog")
    pairs: dict[int, int] = {}
    for identity in sorted(gallery):
        row = by_id[identity]
        if not row["upgraded"]:
            continue
        if row["sourceType"] != "pIdol":
            raise CalibratedPItemReferenceError("Enhanced gallery item must have pIdol source")
        states = groups[(row["sourceType"], row["pIdolId"])]
        if len(states[False]) != 1 or len(states[True]) != 1:
            raise CalibratedPItemReferenceError("Enhanced item requires a unique normal/enhanced catalog pair")
        normal_id = states[False][0]
        if normal_id not in gallery:
            raise CalibratedPItemReferenceError("Normal pair member is missing from the gallery")
        pairs[identity] = normal_id
    return pairs
