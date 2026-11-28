"""Select a local H.I.F. decision algorithm behind the existing business contract."""

from __future__ import annotations

import math
from pathlib import Path

from agent.live_decision.hif_gkms_runtime import (
    UINET21_REV as REV,
)
from agent.live_decision.hif_gkms_runtime import (
    UINET21_ROOT as MODEL_ROOT,
)
from agent.live_decision.hif_gkms_runtime import (
    UINET21_SHA256 as MODEL_SHA256,
)
from agent.live_decision.hif_gkms_runtime import (
    load_uinet21 as load_uinet21,
)
from agent.live_decision.high_score.contracts import PlanType, DecisionKind, DeckKnowledge
from agent.live_decision.high_score.decision_box import DecisionBox as DecisionBox
from agent.live_decision.high_score.decision_box import open_decision_box as open_shared_decision_box
from agent.live_decision.high_score.policy_factory import shared_policy_builder

MASTER = Path(__file__).resolve().parents[1] / 'assets/data/hif_fast/master'


def open_decision_box(profile, *, kernel, opening, planning_seed, deadline, context=None, history=None,
                      model_path=None, model_sha256=None, exam_effects=None):
    """Use one explicit profile with the same choose and completed-step methods."""
    opening.validate()
    if (opening.plan_type is not PlanType.SENSE or opening.ruleset_id != kernel.ruleset_id
            or opening.decision_kind is not DecisionKind.LIVE
            or opening.state.terminal or not opening.capabilities.all_ready
            or opening.state.deck.knowledge is not DeckKnowledge.KNOWN_COMPOSITION
            or opening.state.deck.known_top_count):
        raise ValueError('decision_box_unsupported_opening')
    if not math.isfinite(deadline):
        raise ValueError('decision_box_invalid_deadline')
    if profile not in {'uinet-1.1', 'uinet-2.1', 'strong'} and shared_policy_builder(profile) is None:
        raise ValueError('decision_box_unknown_profile')
    return open_shared_decision_box(profile, kernel=kernel, opening=opening, context=context,
        history=history, planning_seed=planning_seed, deadline=deadline, model_path=model_path,
        model_sha256=model_sha256, exam_effects=exam_effects, use_observed_hand_order=False)
