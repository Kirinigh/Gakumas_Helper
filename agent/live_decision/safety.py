from __future__ import annotations

from dataclasses import dataclass

from .contracts import LiveState, SafeDecision, ContractError, DecisionProposal


@dataclass(frozen=True)
class LiveSafetyConfig:
    provider_timeout_ms: float | None
    max_snapshot_age_ms: float | None
    minimum_identity_confidence: float | None
    allowed_unknown_fields: tuple[str, ...] = ()


class DeterministicLiveSafetyFilter:
    """Fail-closed legality, freshness, resource and identity gate."""

    def __init__(self, config: LiveSafetyConfig):
        self.config = config

    def _stop(self, reason: str, proposal: DecisionProposal | None = None) -> SafeDecision:
        return SafeDecision(status="STOP", reason=reason, action=None, proposal=proposal)

    def filter(self, state: LiveState, proposal: DecisionProposal | None) -> SafeDecision:
        try:
            state.validate()
        except ContractError as error:
            return self._stop(f"invalid_state:{error}", proposal)
        if any(
            threshold is None
            for threshold in (
                self.config.provider_timeout_ms,
                self.config.max_snapshot_age_ms,
                self.config.minimum_identity_confidence,
            )
        ):
            return self._stop("unapproved_runtime_threshold", proposal)
        blocking_unknown = sorted(set(state.unknown_fields) - set(self.config.allowed_unknown_fields))
        if blocking_unknown:
            return self._stop(f"unknown_state:{','.join(blocking_unknown)}", proposal)
        if state.contradictory_fields:
            return self._stop(f"contradictory_state:{','.join(sorted(state.contradictory_fields))}", proposal)
        assert self.config.max_snapshot_age_ms is not None
        if state.capture_age_ms > self.config.max_snapshot_age_ms:
            return self._stop("stale_state", proposal)
        if proposal is None:
            return self._stop("provider_missing")
        if proposal.protocol_version != state.protocol_version:
            return self._stop("protocol_version_mismatch", proposal)
        assert self.config.provider_timeout_ms is not None
        if proposal.elapsed_ms > self.config.provider_timeout_ms:
            return self._stop("provider_timeout", proposal)
        if proposal.status != "PROPOSE" or proposal.action is None:
            return self._stop(f"provider_stop:{proposal.reason}", proposal)
        if proposal.action.kind != "PLAY_CARD" or proposal.action.card_slot is None:
            return self._stop("unsupported_action", proposal)
        matches = [card for card in state.hand if card.slot == proposal.action.card_slot]
        if len(matches) != 1:
            return self._stop("action_card_missing", proposal)
        card = matches[0]
        if not card.playable:
            return self._stop("action_card_not_playable", proposal)
        if card.cost > state.stamina:
            return self._stop("insufficient_stamina", proposal)
        if not card.identity_accepted:
            return self._stop("card_identity_unaccepted", proposal)
        assert self.config.minimum_identity_confidence is not None
        if card.identity_confidence < self.config.minimum_identity_confidence:
            return self._stop("card_identity_below_threshold", proposal)
        if card.box is None or card.box[2] <= 0 or card.box[3] <= 0:
            return self._stop("card_click_target_missing", proposal)
        return SafeDecision(status="ALLOW", reason="all_safety_gates_passed", action=proposal.action, proposal=proposal)
