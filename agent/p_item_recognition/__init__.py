"""P-item image recognition primitives for arena composition reading."""

from .types import UNKNOWN_P_ITEM_ID, PItemHit, PItemPrediction, ResolvedPItemIdentity
from .embedding import (
    OnnxPItemEmbedder,
    PItemEmbeddingGallery,
    PItemEmbeddingRecognizer,
    resolve_p_item_identity,
)
from .reference import (
    PItemCompleteness,
    PItemReferenceHit,
    PItemReferenceError,
    PItemReferenceRuntime,
    PItemReferenceDecision,
    PItemRenderedReferenceGallery,
    PItemReferenceAccelerationProfile,
    measure_p_item_completeness,
    measure_p_item_content_generation,
)
from .preprocess import (
    P_ITEM_ART_BOX,
    P_ITEM_UPGRADE_MARKER_BOX,
    focus_p_item_art,
    extract_p_item_upgrade_marker,
)

__all__ = [
    "P_ITEM_ART_BOX",
    "P_ITEM_UPGRADE_MARKER_BOX",
    "extract_p_item_upgrade_marker",
    "focus_p_item_art",
    "OnnxPItemEmbedder",
    "PItemEmbeddingGallery",
    "PItemEmbeddingRecognizer",
    "resolve_p_item_identity",
    "UNKNOWN_P_ITEM_ID",
    "PItemHit",
    "PItemPrediction",
    "ResolvedPItemIdentity",
    "PItemCompleteness",
    "PItemReferenceAccelerationProfile",
    "PItemReferenceDecision",
    "PItemReferenceError",
    "PItemReferenceHit",
    "PItemReferenceRuntime",
    "PItemRenderedReferenceGallery",
    "measure_p_item_content_generation",
    "measure_p_item_completeness",
]
