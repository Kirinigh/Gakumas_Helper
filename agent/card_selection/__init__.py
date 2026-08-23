"""Safe card-ID recognition and produce reward selection primitives."""

from .types import UNKNOWN, CandidateBox, CardPrediction, SelectionContext, SelectionDecision
from .voting import MultiFrameVoter
from .capture import CaptureEnvironment, CardCaptureAdapter, new_capture_batch_id
from .decision import CardSelectionPolicy, load_policy
from .embedding import (
    CARD_ART_EXCLUSION_BOXES,
    EmbeddingHit,
    EmbeddingGallery,
    OnnxCardEmbedder,
    ResolvedEmbeddingHit,
    EmbeddingCardRecognizer,
    focus_card_art,
    resolve_card_identity,
    extract_upgrade_marker,
    resolve_visual_ambiguity,
)
from .environment import DmmWindowFacts, validate_dmm_window

__all__ = [
    "UNKNOWN",
    "CardCaptureAdapter",
    "CaptureEnvironment",
    "new_capture_batch_id",
    "CardPrediction",
    "CandidateBox",
    "CardSelectionPolicy",
    "CARD_ART_EXCLUSION_BOXES",
    "DmmWindowFacts",
    "EmbeddingGallery",
    "EmbeddingCardRecognizer",
    "EmbeddingHit",
    "MultiFrameVoter",
    "OnnxCardEmbedder",
    "ResolvedEmbeddingHit",
    "SelectionContext",
    "SelectionDecision",
    "focus_card_art",
    "extract_upgrade_marker",
    "load_policy",
    "resolve_card_identity",
    "resolve_visual_ambiguity",
    "validate_dmm_window",
]
