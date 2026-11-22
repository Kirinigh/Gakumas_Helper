"""Business continuity for opt-in, completely observed current-hand support.

This module grants no UI authority. The native source verifier must first
reconstruct the ordinary source, then recheck the full receipt and the actual
transition. Observed component fields alone cannot establish these effects.
"""
from __future__ import annotations

from .hif_support_planning import clean_hand_support


def hand_support_candidate_continuity(before, after):
    """Necessary candidate checks; full kernel transition is still required.

    This alone cannot prove absence of leave-and-redraw. The producer supplies
    actual kernel outcomes; persisted consumers recompute those outcomes.
    """
    return (clean_hand_support(before) == before and clean_hand_support(after) == after
            and set(after.hand_support).issubset(before.hand_support))


def observed_hand_variant_key(state, card):
    """A supported and an ordinary copy are not physically interchangeable."""
    if not state.hand_support:
        return card.semantic_key
    support = tuple((fact.enhancement_count, fact.projection_kind) for fact in state.hand_support
                    if fact.instance_id == card.instance_id)
    return card.semantic_key, support
