"""Family-only artwork continuity from a fully verified untouched opening.

Original identity and current pixels are separate evidence. This never reads a
new title, resolves a plus tier, or supplies an input deadline.
"""
import math
from dataclasses import replace, dataclass


@dataclass(frozen=True)
class _CurrentIdentity:
    frame_id: str
    slot: int
    instance_id: str
    card_id: str
    upgraded: bool
    confidence: float
    box: tuple
    state: object


def restore_reference(source, bundle, opening):
    import cv2
    from route.engine import has_single_frame_id
    from card_selection.model import frame_identifier

    core = opening.get('core', {})
    ids = tuple(core.get('frame_ids', ()))
    hand = bundle.state.deck.hand
    families = tuple(tuple(f) for f in core.get('families', ()))
    if (opening.get('accepted') is not True or opening.get('strategy_state_bound') is not True
            or opening.get('executed') is not False or opening.get('execution_submit_count') != 0
            or (opening.get('run_id'), opening.get('revision')) != (bundle.run_id, bundle.revision)
            or bundle.state.turn != 1 or bundle.state.score != 0
            or bundle.state.deck.discard or bundle.state.deck.removed
            or len(hand) not in (4, 5) or len(families) != len(hand)
            or not has_single_frame_id(ids)):
        return None
    for card, family in zip(hand, families, strict=True):
        definition = bundle.kernel.cards[card.card_id]
        if (definition.canonical_card_id or card.card_id) not in family:
            return None
    paths = {f['frame_id']: f.get('path') for f in source.get('frames', ())}
    fid, = ids
    if not paths.get(fid):
        return None
    image = cv2.imread(paths[fid])
    if image is None or frame_identifier(image) != fid:
        return None
    image.flags.writeable = False
    return dict(run_id=bundle.run_id, revision=bundle.revision, turn=bundle.state.turn,
        source=source['id'], frame_ids=ids, images=(image,), families=families,
        hand_instance_ids=tuple(c.instance_id for c in hand), state=bundle.state,
        action_order=tuple(core['fields'][5]))


def corroborate(context, image, candidates, predictions, *, selected_slot, evidence):
    from card_selection.model import frame_identifier

    from .hif_visual import hif_unselected_row_candidates
    from .hif_inspected_art import LEFT_HALF_PROFILE, visible_art

    reference = getattr(context, 'gkh_hif_opening_art_reference', None)
    bundle = getattr(context, 'gkh_hif_opening_state', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    committed = getattr(context, 'gkh_hif_committed_state', None)
    if (not reference or bundle is None or run is None
            or (run.run_id, run.revision) != (reference['run_id'], reference['revision'])
            or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)
            or bundle.state != reference['state']
            or committed is not None and committed != reference['state']
            or tuple(c.instance_id for c in bundle.state.deck.hand) != reference['hand_instance_ids']
            or len(candidates) != len(predictions) or len(candidates) != len(reference['families'])):
        return tuple(predictions)
    fid = frame_identifier(image)
    current_proofs = getattr(context, 'gkh_hif_opening_current_art', None)
    if selected_slot is not None or not current_proofs or current_proofs[0] != fid:
        current_proofs = (fid, {})
        context.gkh_hif_opening_current_art = current_proofs
    if fid in reference['frame_ids']:
        return tuple(predictions)
    result = list(predictions)
    for slot, prediction in enumerate(predictions):
        if (selected_slot is not None or slot >= len(predictions)-1
                or prediction.slot != slot or prediction.frame_id != fid or prediction.accepted
                or prediction.recognizer != 'card_art_embedding_top5_hif_masked_v1'
                or prediction.reason not in ('below_acceptance_threshold', 'below_margin_threshold')
                or not all(math.isfinite(v) and 0 <= v <= 1 for v in (prediction.confidence, prediction.margin))
                or not prediction.top_k_card_ids
                or tuple(prediction.top_k_card_ids[0]) != reference['families'][slot]):
            continue
        # The original row is unselected. A current selected neighbor is not
        # allowed to redefine the source footprint or its compared region.
        try:
            if hif_unselected_row_candidates(image.shape, candidates, image=image) != tuple(candidates):
                continue
            if any(hif_unselected_row_candidates(old.shape, candidates, image=old) != tuple(candidates)
                    for old in reference['images']):
                continue
            current = visible_art(image, candidates, slot, visibility_profile=LEFT_HALF_PROFILE)
            if any(visible_art(old, candidates, slot, visibility_profile=LEFT_HALF_PROFILE) != current
                    for old in reference['images']):
                continue
        except ValueError:
            continue
        # Preserve the network score and ranking. The additional proof is exact
        # current artwork, not a raised confidence or a newly read full title.
        evidence.append(dict(scope='verified_opening_exact_current_art_v1', accepted=True,
            slot=slot, source=reference['source'], reference_frame_ids=reference['frame_ids'],
            frame_id=fid, family=reference['families'][slot],
            original_prediction=prediction.to_dict(), pixel_comparison='exact_fixed_left_half_circle',
            upgrade_verified=False, input_authorized=False))
        result[slot] = replace(prediction, accepted=True, reason='verified_opening_exact_current_art')
        card = bundle.state.deck.hand[slot]
        definition = bundle.kernel.cards[card.card_id]
        current_proofs[1][slot] = _CurrentIdentity(fid, slot, card.instance_id,
            definition.canonical_card_id or card.card_id, card.upgraded,
            prediction.confidence, tuple(prediction.box), bundle.state)
    return tuple(result)


def safety_identity_verified(context, state, card):
    """Consume only an in-process proof produced from this exact current frame.

    Serialized accepted flags, the old opening receipt, and an old proof cannot
    enable this path. The public card keeps its original neural score; the
    ordinary safety filter still checks uncertainty, age, legality and cost.
    """
    reference = getattr(context, 'gkh_hif_opening_art_reference', None)
    current = getattr(context, 'gkh_hif_opening_current_art', None)
    bundle = getattr(context, 'gkh_hif_opening_state', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    if (not reference or not current or current[0] != state.snapshot_id or bundle is None or run is None
            or (run.run_id, run.revision) != (reference['run_id'], reference['revision'])
            or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)
            or bundle.state != reference['state']
            or getattr(context, 'gkh_hif_committed_state', None) not in (None, reference['state'])):
        return False
    proof = current[1].get(card.slot)
    if (type(proof) is not _CurrentIdentity or proof.state != bundle.state
            or (proof.frame_id, proof.slot) != (state.snapshot_id, card.slot)
            or not 0 <= card.slot < len(reference['hand_instance_ids'])
            or (card.card_id, card.upgraded, card.box)
                != (proof.card_id, proof.upgraded, proof.box)
            or proof.instance_id != reference['hand_instance_ids'][card.slot]
            or len(state.hand) != len(bundle.state.deck.hand)):
        return False
    expected = bundle.state
    if ((state.turn, state.turns_remaining, state.stamina, state.max_stamina, state.score, state.score_multiplier,
            tuple(state.action_order)) != (expected.turn, expected.turns_remaining, expected.stamina,
            expected.max_stamina, expected.score, expected.score_multiplier, reference['action_order'])):
        return False
    for slot, (observed, held) in enumerate(zip(state.hand, expected.deck.hand, strict=True)):
        definition = bundle.kernel.cards[held.card_id]
        if (observed.slot, observed.card_id, observed.upgraded) != (
                slot, definition.canonical_card_id or held.card_id, held.upgraded):
            return False
    return True
