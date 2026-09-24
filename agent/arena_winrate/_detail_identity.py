"""Pure checks for detail identity evidence; no capture, state store or input."""
from __future__ import annotations

from typing import Any, NamedTuple
from collections.abc import Callable, Sequence

from .reader import ClickedSkillCard
from ._reader_evidence import _DetailIdentityProof, _DetailIdentityObservation


class DetailIdentityTransaction(NamedTuple):
    """A shallow view of the current opening; frame identity must be preserved."""

    transaction_started: float | None
    transaction_token: int | None
    source_card_box: tuple[int, int, int, int] | None
    interaction_box: tuple[int, int, int, int] | None
    contact_released_at: float | None
    capture_started_at: float | None
    detail_image: Any
    source_guard_frames: Any
    restoration_signatures: Any
    identity_frames: Any


def build_detail_identity_proof(
    resolved: ClickedSkillCard,
    observations: Sequence[_DetailIdentityObservation],
    titles_match: Callable[[str, str, int], bool],
) -> _DetailIdentityProof | None:
    """Validate the existing last-two-observation contract without publishing it."""
    selected = tuple(observations[-2:])
    if not selected:
        return None
    first, last = selected[0], selected[-1]
    expected_customizations = tuple(sorted(
        (str(customization_id), int(count))
        for customization_id, count in resolved.customizations.items()
    ))
    shared_fields_match = all(
        observation.transaction_token == first.transaction_token
        and observation.transaction_started == first.transaction_started
        and observation.source_card_box == first.source_card_box
        and observation.interaction_box == first.interaction_box
        and observation.contact_released_at == first.contact_released_at
        and observation.source_guard_frames is first.source_guard_frames
        and observation.restoration_signatures is first.restoration_signatures
        and observation.identity_frames is first.identity_frames
        and titles_match(observation.title, first.title, resolved.card_id)
        and observation.card_id == first.card_id == resolved.card_id
        for observation in selected
    )
    captures = tuple(observation.capture_started_at for observation in selected)
    images = tuple(observation.detail_image for observation in selected)
    fresh = bool(
        first.contact_released_at < captures[0]
        and all(earlier < later for earlier, later in zip(captures, captures[1:], strict=False))
        and len({id(image) for image in images}) == len(images)
    )
    if not (shared_fields_match and fresh):
        return None
    return _DetailIdentityProof(
        transaction_token=last.transaction_token,
        transaction_started=last.transaction_started,
        source_card_box=last.source_card_box,
        interaction_box=last.interaction_box,
        contact_released_at=last.contact_released_at,
        capture_started_at=captures,
        detail_images=images,
        source_guard_frames=last.source_guard_frames,
        restoration_signatures=last.restoration_signatures,
        identity_frames=last.identity_frames,
        title=last.title,
        card_id=last.card_id,
        customizations=expected_customizations,
        resolution_source=resolved.resolution_source,
        evidence_mode=resolved.detail_evidence_mode,
        detail_confirmation_reads=resolved.detail_confirmation_reads,
    )


def matches_detail_identity_transaction(
    proof: _DetailIdentityProof,
    current: DetailIdentityTransaction,
) -> bool:
    """Compare live transaction values and exact source objects, never copies."""
    return bool(
        current.transaction_started == proof.transaction_started
        and current.transaction_token == proof.transaction_token
        and current.source_card_box == proof.source_card_box
        and current.interaction_box == proof.interaction_box
        and current.contact_released_at == proof.contact_released_at
        and current.capture_started_at == proof.capture_started_at[-1]
        and current.detail_image is proof.detail_images[-1]
        and current.source_guard_frames is proof.source_guard_frames
        and current.restoration_signatures is proof.restoration_signatures
        and current.identity_frames is proof.identity_frames
    )


def detail_identity_proof_matches(
    proof: _DetailIdentityProof,
    expected_card_id: int,
    source_card_box: tuple[int, int, int, int],
    current: DetailIdentityTransaction,
) -> bool:
    """Check an existing proof's identity, freshness and active transaction."""
    captures, images = proof.capture_started_at, proof.detail_images
    fresh = bool(
        captures
        and len(captures) == len(images)
        and proof.contact_released_at < captures[0]
        and all(earlier < later for earlier, later in zip(captures, captures[1:], strict=False))
        and len({id(image) for image in images}) == len(images)
    )
    return bool(
        proof.card_id == expected_card_id
        and proof.source_card_box == tuple(source_card_box)
        and proof.title.strip()
        and fresh
        and matches_detail_identity_transaction(proof, current)
    )


def detail_identity_proof_diagnostic(proof: _DetailIdentityProof) -> dict[str, Any]:
    """Return the unchanged image-free provenance fields."""
    return {
        "transaction_token": proof.transaction_token,
        "transaction_started": proof.transaction_started,
        "source_card_box": list(proof.source_card_box),
        "interaction_box": list(proof.interaction_box),
        "contact_released_at": proof.contact_released_at,
        "capture_started_at": list(proof.capture_started_at),
        "title": proof.title,
        "card_id": proof.card_id,
        "customizations": dict(proof.customizations),
        "resolution_source": proof.resolution_source,
        "evidence_mode": proof.evidence_mode,
        "detail_confirmation_reads": proof.detail_confirmation_reads,
    }
