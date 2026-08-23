from __future__ import annotations

from typing import Any
from collections import deque

P_ITEM_REFERENCE_SIZE = 130
P_ITEM_ART_SIZE = 108
P_ITEM_ART_BOX = (11, 11, 119, 119)
P_ITEM_UPGRADE_MARKER_BOX = (104, 0, 130, 26)
P_ITEM_ART_MARKER_EXCLUSION_BOX = (90, 0, 108, 18)
P_ITEM_BACKGROUND_MINIMUM = 168
P_ITEM_BACKGROUND_MAX_CHANNEL_SPREAD = 92
P_ITEM_DARK_FRAME_REFERENCE_CUTOFF = 220


def _as_pil_image(image: Any) -> Any:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for P-item preprocessing") from error

    return image if isinstance(image, Image.Image) else Image.fromarray(image)


def _normalise_reference_size(image: Any) -> Any:
    from PIL import Image

    source = _as_pil_image(image).convert("RGB")
    if source.size != (P_ITEM_REFERENCE_SIZE, P_ITEM_REFERENCE_SIZE):
        source = source.resize(
            (P_ITEM_REFERENCE_SIZE, P_ITEM_REFERENCE_SIZE), Image.Resampling.LANCZOS
        )
    return source


def focus_p_item_art(image: Any) -> Any:
    """Keep the centred item artwork while excluding its frame and top-right + marker."""

    import numpy as np
    from PIL import Image, ImageDraw

    focused = _normalise_reference_size(image).crop(P_ITEM_ART_BOX)
    draw = ImageDraw.Draw(focused)
    draw.rectangle(P_ITEM_ART_MARKER_EXCLUSION_BOX, fill=(255, 255, 255))
    values = np.asarray(focused, dtype=np.uint8).copy()
    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    brightness_reference = float(np.percentile(maximum, 95))
    eligibility_scale = (
        255.0 / max(brightness_reference, 1.0)
        if brightness_reference < P_ITEM_DARK_FRAME_REFERENCE_CUTOFF
        else 1.0
    )
    eligible = (minimum.astype(np.float32) * eligibility_scale >= P_ITEM_BACKGROUND_MINIMUM) & (
        (maximum.astype(np.float32) - minimum.astype(np.float32)) * eligibility_scale
        <= P_ITEM_BACKGROUND_MAX_CHANNEL_SPREAD
    )
    height, width = eligible.shape
    connected = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        queue.append((0, x))
        queue.append((height - 1, x))
    for y in range(1, height - 1):
        queue.append((y, 0))
        queue.append((y, width - 1))
    while queue:
        y, x = queue.popleft()
        if connected[y, x] or not eligible[y, x]:
            continue
        connected[y, x] = True
        if y:
            queue.append((y - 1, x))
        if y + 1 < height:
            queue.append((y + 1, x))
        if x:
            queue.append((y, x - 1))
        if x + 1 < width:
            queue.append((y, x + 1))
    values[connected] = 255
    return Image.fromarray(values, mode="RGB")


def extract_p_item_upgrade_marker(image: Any) -> Any:
    """Return the isolated top-right normal/+ state marker as a stable RGB tensor."""

    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("numpy and Pillow are required for P-item state preprocessing") from error

    marker = _normalise_reference_size(image).crop(P_ITEM_UPGRADE_MARKER_BOX).resize(
        (24, 24), Image.Resampling.BILINEAR
    )
    return np.asarray(marker, dtype=np.float32) / 255.0
