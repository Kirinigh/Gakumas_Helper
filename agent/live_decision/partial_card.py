from __future__ import annotations

import statistics
from typing import Any, Iterable
from dataclasses import replace, dataclass

try:
    from agent.card_selection.types import UNKNOWN, CardPrediction
    from agent.card_selection.embedding import focus_card_art
except ModuleNotFoundError:  # Maa runtime exposes ``agent/`` as the import root.
    from card_selection.types import UNKNOWN, CardPrediction
    from card_selection.embedding import focus_card_art

MIN_LIVE_HAND_SIZE = 1
MAX_LIVE_HAND_SIZE = 5
MIN_SUPPORTED_VISIBLE_RATIO = 0.55
MIN_HIF_MASK_VISIBLE_RATIO = 0.25
MIN_LIVE_HAND_TOP_RATIO = 0.55


class LiveCardLayoutError(ValueError):
    """Raised when Live card geometry cannot be reconstructed safely."""


@dataclass(frozen=True)
class LivePartialCardView:
    """One card-art crop with its occluded width restored as a white mask."""

    slot: int
    candidate_box: tuple[int, int, int, int]
    visible_box: tuple[int, int, int, int]
    canvas_size: tuple[int, int]
    visible_ratio: float
    view_mode: str
    layout_version: str
    canvas_offset: tuple[int, int] = (0, 0)


def _hif_overlapping_card_views(frame_shape, cards, selected_slot):
    """Recover the row footprint, including occlusion by a raised foreground card.

    SELECT is supplied only after the caller's existing three-frame consensus.
    Detector boxes describe visible fragments, not the underlying card width.
    The selected card's right neighbour may therefore have a shifted left edge.
    """
    count = len(cards)
    if frame_shape[:2] != (1280, 720) or count not in (4, 5):
        raise LiveCardLayoutError("HIF overlapping card art layout is uncalibrated")
    if selected_slot is not None and (type(selected_slot) is not int or not 0 <= selected_slot < count):
        raise LiveCardLayoutError("HIF selected slot is invalid")
    covered_slot = selected_slot + 1 if selected_slot is not None and selected_slot < count - 1 else None
    stable = [i for i in range(count) if i != covered_slot]
    ordinary = [i for i in stable if i != selected_slot]
    row_top = statistics.median(cards[i].box[1] for i in ordinary)
    row_height = statistics.median(cards[i].box[3] for i in ordinary)
    if not (878 <= row_top <= 890 and 240 <= row_height <= 256):
        raise LiveCardLayoutError("HIF card row dimensions are uncalibrated")
    if any(abs(cards[i].box[1] - row_top) > 6 or abs(cards[i].box[3] - row_height) > 10
           for i in ordinary):
        raise LiveCardLayoutError("HIF ordinary card row is unstable")
    if selected_slot is not None:
        selected = cards[selected_slot]
        if not (18 <= row_top - selected.box[1] <= 34
                and abs(selected.box[3] - row_height) <= 12):
            raise LiveCardLayoutError("HIF selected card geometry disagrees with SELECT")
    spacing = statistics.median((cards[j].box[0] - cards[i].box[0]) / (j - i)
                                for i in stable for j in stable if j > i)
    full_width = 190 if count == 4 else round(row_height * 0.765)
    art_size = 180 if count == 4 else round(full_width * 0.947)
    inset = round((full_width - art_size) / 2)
    origin = statistics.median(cards[i].box[0] - i * spacing for i in stable)
    low, high = (.80, .90) if count == 4 else (.60, .72)
    if not low <= spacing / full_width <= high:
        raise LiveCardLayoutError("HIF card spacing is uncalibrated")
    anchors = [round(origin + i * spacing) for i in range(count)]
    if any(abs(cards[i].box[0] - anchors[i]) > 8 for i in stable):
        raise LiveCardLayoutError("HIF card anchors are inconsistent")
    if covered_slot is not None and not -8 <= cards[covered_slot].box[0] - anchors[covered_slot] <= full_width * 0.4:
        raise LiveCardLayoutError("HIF occluded neighbour anchor is inconsistent")
    views = []
    for i, card in enumerate(cards):
        art_left = anchors[i] + inset
        left = art_left
        right = art_left + art_size
        if i != selected_slot:
            if i < count - 1:
                right = min(right, anchors[i + 1] - 2)
            if i == covered_slot:
                left = max(left, anchors[selected_slot] + full_width + 2)
        top = round((card.box[1] if i == selected_slot else row_top) + row_height * 0.06)
        box = (left, top, right - left, art_size)
        _validate_box(box, frame_shape[1], frame_shape[0])
        ratio = box[2] / art_size
        views.append(LivePartialCardView(
            slot=i, candidate_box=card.box, visible_box=box,
            canvas_size=(art_size, art_size), visible_ratio=ratio,
            view_mode="left_occluded" if left > art_left else _view_mode(ratio),
            layout_version="hif-four-card-art-v1" if count == 4 else "hif-five-card-art-v2",
            canvas_offset=(left - art_left, 0),
        ))
    return tuple(views)


def normalise_live_candidates(
    candidates: Iterable[Any],
    *,
    frame_shape: tuple[int, ...] | None = None,
) -> tuple[Any, ...]:
    """Sort and deduplicate 1..5 Live candidates without silently truncating them."""

    ordered = sorted(candidates, key=lambda candidate: (candidate.box[0], candidate.box[1]))
    if frame_shape is not None:
        if len(frame_shape) < 2:
            return ()
        frame_height, frame_width = frame_shape[:2]
        if frame_height <= 0 or frame_width <= 0:
            return ()
        min_live_top = frame_height * MIN_LIVE_HAND_TOP_RATIO
        ordered = [
            candidate
            for candidate in ordered
            if candidate.box[0] >= 0
            and candidate.box[1] >= min_live_top
            and candidate.box[0] + candidate.box[2] <= frame_width
            and candidate.box[1] + candidate.box[3] <= frame_height
        ]
    deduplicated: list[Any] = []
    for candidate in ordered:
        x, _y, width, _height = candidate.box
        if width <= 0:
            continue
        centre = x + width / 2.0
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(deduplicated)
                if abs(centre - (existing.box[0] + existing.box[2] / 2.0))
                <= 0.18 * min(width, existing.box[2])
            ),
            None,
        )
        if duplicate_index is None:
            deduplicated.append(candidate)
        elif candidate.detector_score > deduplicated[duplicate_index].detector_score:
            deduplicated[duplicate_index] = candidate
    if not MIN_LIVE_HAND_SIZE <= len(deduplicated) <= MAX_LIVE_HAND_SIZE:
        return ()
    deduplicated.sort(key=lambda candidate: (candidate.box[0], candidate.box[1]))
    return tuple(replace(candidate, slot=slot) for slot, candidate in enumerate(deduplicated))


def _view_mode(visible_ratio: float) -> str:
    if visible_ratio >= 0.90:
        return "full"
    return min((0.55, 0.65, 0.75), key=lambda value: abs(value - visible_ratio)).__format__(
        ".2f"
    )


def _validate_box(
    box: tuple[int, int, int, int], frame_width: int, frame_height: int
) -> None:
    x, y, width, height = box
    if (
        x < 0
        or y < 0
        or width < 8
        or height < 8
        or x + width > frame_width
        or y + height > frame_height
    ):
        raise LiveCardLayoutError("Live partial-card ROI is outside the frame")


def build_live_card_views(
    frame_shape: tuple[int, ...],
    candidates: Iterable[Any],
    *,
    legacy_art_boxes: Iterable[tuple[int, int, int, int]] | None = None,
    layout_profile: str | None = None,
    selected_slot: int | None = None,
) -> tuple[LivePartialCardView, ...]:
    """Reconstruct canonical card-art views for an unoccluded or overlapping hand."""

    if len(frame_shape) < 2:
        raise LiveCardLayoutError("Live frame shape is invalid")
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    cards = normalise_live_candidates(candidates)
    if not cards:
        raise LiveCardLayoutError("Live hand must contain one to five unique cards")
    if layout_profile is not None:
        profiles = {"hif-five-card-art-v2": 5, "hif-four-card-art-v1": 4}
        if profiles.get(layout_profile) != len(cards) or legacy_art_boxes:
            raise LiveCardLayoutError("Live card art profile is invalid or conflicts with legacy boxes")
        return _hif_overlapping_card_views(frame_shape, cards, selected_slot)
    if selected_slot is not None:
        raise LiveCardLayoutError("selected art requires an explicit calibrated layout profile")

    legacy = tuple(legacy_art_boxes or ())
    if legacy:
        if len(legacy) != len(cards):
            raise LiveCardLayoutError("legacy Live art boxes do not match the hand")
        views = []
        for card, box in zip(cards, legacy, strict=True):
            _validate_box(box, frame_width, frame_height)
            views.append(
                LivePartialCardView(
                    slot=card.slot,
                    candidate_box=card.box,
                    visible_box=box,
                    canvas_size=(box[2], box[3]),
                    visible_ratio=1.0,
                    view_mode="full",
                    layout_version="dmm-live-legacy-art-v1",
                )
            )
        return tuple(views)

    if len(cards) <= 3:
        views = []
        for card in cards:
            x, y, width, height = card.box
            art_width = round(width * 0.95)
            art_height = min(art_width, round(height * 0.74))
            box = (
                x + round(width * 0.025),
                y + round(height * 0.06),
                art_width,
                art_height,
            )
            _validate_box(box, frame_width, frame_height)
            views.append(
                LivePartialCardView(
                    slot=card.slot,
                    candidate_box=card.box,
                    visible_box=box,
                    canvas_size=(art_width, art_height),
                    visible_ratio=1.0,
                    view_mode="full",
                    layout_version="dmm-live-unoccluded-v1",
                )
            )
        return tuple(views)

    anchors = [card.box[0] for card in cards]
    spacings = [right - left for left, right in zip(anchors, anchors[1:])]
    spacing = float(statistics.median(spacings))
    if spacing < 24 or any(abs(value - spacing) > max(6.0, spacing * 0.28) for value in spacings):
        raise LiveCardLayoutError("Live overlapping hand spacing is inconsistent")

    if len(cards) == 4:
        # Recommendation/selection animation can transiently enlarge one
        # detector box.  Canonical four-card crops must follow the stable row,
        # not that outlier.
        detector_width = round(statistics.median(card.box[2] for card in cards))
        row_top = round(statistics.median(card.box[1] for card in cards))
        full_width = min(
            max(detector_width, round(spacing * 1.05)),
            round(spacing * 1.35),
        )
        art_top = row_top + round(full_width * 0.13)
        art_height = full_width
        horizontal_inset = round(full_width * 0.05)
        layout_version = "dmm-live-overlap-four-v1"
    else:
        maximum_detector_width = max(card.box[2] for card in cards)
        row_top = min(card.box[1] for card in cards)
        full_width = round(statistics.median((maximum_detector_width, spacing * 1.38)))
        full_width = min(max(full_width, round(spacing * 1.20)), round(spacing * 1.60))
        art_top = row_top + round(spacing * 0.15)
        art_height = round(spacing * 1.40)
        horizontal_inset = round(spacing * 0.075)
        layout_version = "dmm-live-overlap-v1"
    neighbour_gap = max(2, round(spacing * 0.02))

    views = []
    for index, card in enumerate(cards):
        left = card.box[0] + horizontal_inset
        right = (
            anchors[index + 1] - neighbour_gap
            if index + 1 < len(cards)
            else min(card.box[0] + full_width, frame_width)
        )
        visible_width = right - left
        visible_ratio = visible_width / full_width
        if visible_ratio < MIN_SUPPORTED_VISIBLE_RATIO or visible_ratio > 1.05:
            raise LiveCardLayoutError(
                f"Live card slot {card.slot} has unsupported visible ratio {visible_ratio:.3f}"
            )
        box = (left, art_top, visible_width, art_height)
        _validate_box(box, frame_width, frame_height)
        views.append(
            LivePartialCardView(
                slot=card.slot,
                candidate_box=card.box,
                visible_box=box,
                canvas_size=(full_width, art_height),
                visible_ratio=min(visible_ratio, 1.0),
                view_mode=_view_mode(min(visible_ratio, 1.0)),
                layout_version=layout_version,
            )
        )
    return tuple(views)


def preprocess_live_card_view(
    image: Any,
    view: LivePartialCardView,
    *,
    source_color_order: str,
    exclusion_boxes: tuple[tuple[float, float, float, float], ...] = (),
) -> Any:
    """Return one canonical 64x64 RGB tensor without stretching visible fragments."""

    import numpy as np
    from PIL import Image, ImageDraw

    if source_color_order not in {"RGB", "BGR"}:
        raise ValueError("source_color_order must be RGB or BGR")
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] not in {3, 4}:
        raise ValueError("Live frame must be HxWx3 or HxWx4")
    x, y, width, height = view.visible_box
    _validate_box(view.visible_box, frame.shape[1], frame.shape[0])
    crop = frame[y : y + height, x : x + width, :3]
    if source_color_order == "BGR":
        crop = crop[:, :, ::-1]
    canvas = Image.new("RGB", view.canvas_size, "white")
    ox, oy = view.canvas_offset
    if min(ox, oy) < 0 or ox + width > view.canvas_size[0] or oy + height > view.canvas_size[1]:
        raise LiveCardLayoutError("Live visible fragment exceeds its canonical canvas")
    canvas.paste(Image.fromarray(np.ascontiguousarray(crop, dtype=np.uint8), mode="RGB"), (ox, oy))
    focused = focus_card_art(canvas)
    draw = ImageDraw.Draw(focused)
    for box in exclusion_boxes:
        if len(box) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                    and 0 <= v <= 1 for v in box):
            raise LiveCardLayoutError("Live artwork exclusion is invalid")
        left, top, right, bottom = box
        if left >= right or top >= bottom:
            raise LiveCardLayoutError("Live artwork exclusion is empty")
        width, height = focused.size
        draw.rectangle((round(left * width), round(top * height),
                        round(right * width), round(bottom * height)), fill="white")
    focused = focused.resize((64, 64), Image.Resampling.BILINEAR)
    rgb = np.asarray(focused, dtype=np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None, ...]


def normalize_hif_disabled_art(image, cards, views, *, selected_slot=None):
    """Undo only an independently proven half-dark layer on visible artwork."""
    import numpy as np

    from .hif_visual import (
        hif_selected_row_proposal,
        _hif_disabled_title_plates,
        _hif_unselected_title_plates,
        _hif_selected_disabled_fragments,
        _hif_selected_disabled_title_plates,
    )

    if (getattr(image, 'shape', ()) != (1280,720,3) or image.dtype != np.uint8
            or len(cards) not in (3,4,5) or len(views) != len(cards)
            or tuple(card.slot for card in cards) != tuple(range(len(cards)))
            or any(card.box[2:] != (190,250) for card in cards)):
        return image, ()
    white = _hif_unselected_title_plates(image)
    selected_disabled = ()
    if selected_slot is None:
        disabled = _hif_disabled_title_plates(image,cards,white)
    else:
        if type(selected_slot) is not int or not 0 <= selected_slot < len(cards):
            return image, ()
        proposal = hif_selected_row_proposal(image, cards, hand_count=len(cards))
        if (proposal is None or proposal[1] != selected_slot
                or tuple(card.box for card in proposal[0]) != tuple(card.box for card in cards)):
            return image, ()
        disabled = _hif_selected_disabled_fragments(image,
            tuple(card.box[0] for card in cards), selected_slot, cards[selected_slot].box, white,
            candidates=cards)
        selected_disabled = tuple(plate for plate in _hif_selected_disabled_title_plates(image)
            if (plate[0]-2,plate[1]+plate[3]+2-250,190,250) == cards[selected_slot].box)
        disabled = (*disabled,*selected_disabled)
    if (not white or not disabled
            or len(white)+len(disabled)+(selected_slot is not None and not selected_disabled) != len(cards)):
        return image, ()

    def plate_level(plate):
        x,y,w,h = plate
        colors,counts = np.unique(image[y:y+h,x:x+w].reshape(-1,3),axis=0,return_counts=True)
        best = np.flatnonzero(counts == counts.max())
        if len(best) != 1 or np.ptp(colors[best[0]]) != 0:
            return None
        return int(colors[best[0],0])

    levels = {plate_level(plate) for plate in white}
    if len(levels) != 1 or None in levels:
        return image, ()
    white_level = levels.pop()
    if white_level < 235:
        return image, ()
    proofs = []
    for plate in disabled:
        owners = [card for card in cards if (card.detector_label == 'useless'
                  or plate in selected_disabled and card.detector_label == 'hif_gray_observation')
                  and card.clickable is False
                  and (selected_slot is not None or card.detector_score >= .95
                       or 180 <= plate[2] <= 190 and 0 < card.detector_score <= 1)
                  and abs(card.box[0]-(plate[0]-2)) <= 8]
        exposed_plate = plate
        if selected_slot is not None and plate not in selected_disabled:
            x,y,w,h = plate
            left,_,width,_ = cards[selected_slot].box
            if x < left:
                exposed_plate = (x,y,min(x+w,left-2)-x,h)
            else:
                exposed_plate = (max(x,left+width),y,x+w-max(x,left+width),h)
            if exposed_plate[2] <= 0:
                return image, ()
        gray_level = plate_level(exposed_plate)
        if len(owners) != 1 or gray_level is None or gray_level*2 != white_level:
            return image, ()
        card = owners[0]
        view = views[card.slot]
        if view.slot != card.slot or view.candidate_box != card.box:
            return image, ()
        proofs.append(dict(slot=card.slot,profile='observed_disabled_half_intensity_v1',
            title_plate=plate,white_title_plates=white,gray_level=gray_level,
            white_level=white_level,visible_box=view.visible_box,gain=2))
        if selected_slot is not None:
            proofs[-1].update(selected_slot=selected_slot, exposed_title_plate=exposed_plate)
    if len({proof['slot'] for proof in proofs}) != len(proofs):
        return image, ()
    corrected = image.copy()
    for proof in proofs:
        x,y,w,h = proof['visible_box']
        corrected[y:y+h,x:x+w] = np.minimum(image[y:y+h,x:x+w].astype(np.uint16)*2,255).astype(np.uint8)
    return corrected, tuple(proofs)


def classify_live_card_views(
    recognizer: Any,
    image: Any,
    candidates: Iterable[Any],
    *,
    frame_id: str,
    plan: str,
    acceptance_threshold: float | None,
    min_margin: float | None,
    legacy_art_boxes: Iterable[tuple[int, int, int, int]] | None = None,
    layout_profile: str | None = None,
    selected_slot: int | None = None,
    prepared_art: Any | None = None,
    refine_hif_title_anchor: bool = False,
    hif_full_art_mask: bool = False,
    art_preprocessing_evidence: list | None = None,
) -> tuple[Any, ...]:
    """Return visual-family predictions; Live upgrade resolution remains external."""

    cards = normalise_live_candidates(candidates)
    views = build_live_card_views(
        tuple(image.shape), cards, legacy_art_boxes=legacy_art_boxes,
        layout_profile=layout_profile, selected_slot=selected_slot,
    )
    predictions = []
    corrected, corrections = image, ()
    supported = getattr(recognizer, 'gkh_hif_trial_existing_art_families', ())
    if supported and (layout_profile in ('hif-five-card-art-v2','hif-four-card-art-v1')
                      or hif_full_art_mask and len(cards) == 3):
        normalization_selected = selected_slot
        if hif_full_art_mask and len(cards) == 3 and selected_slot is None:
            from .hif_visual import hif_selected_row_proposal

            proposal = hif_selected_row_proposal(image,cards,hand_count=3)
            if proposal and tuple(c.box for c in proposal[0]) == tuple(c.box for c in cards):
                normalization_selected = proposal[1]
        corrected, corrections = normalize_hif_disabled_art(image,cards,views,
            selected_slot=normalization_selected)
    corrections = {proof['slot']:proof for proof in corrections}
    for card, view in zip(cards, views, strict=True):
        hif_profile = layout_profile in {"hif-five-card-art-v2", "hif-four-card-art-v1"}
        if hif_profile and prepared_art is not None and view.visible_ratio >= MIN_HIF_MASK_VISIBLE_RATIO:
            if prepared_art.recognizer is not recognizer:
                raise LiveCardLayoutError("HIF masked gallery belongs to a different recognizer")
            try:
                prediction = prepared_art.classify(image, view, card, frame_id=frame_id, plan=plan,
                    acceptance_threshold=acceptance_threshold, min_margin=min_margin)
            except ValueError as error:
                raise LiveCardLayoutError("HIF artwork mask is not prepared") from error
            if card.slot in corrections:
                normalized = prepared_art.classify(corrected,view,card,frame_id=frame_id,plan=plan,
                    acceptance_threshold=acceptance_threshold,min_margin=min_margin)
                proof = dict(corrections[card.slot],frame_id=frame_id,
                    original_prediction=prediction.to_dict(),normalized_prediction=normalized.to_dict(),
                    source_families=supported)
                if art_preprocessing_evidence is not None:
                    art_preprocessing_evidence.append(proof)
                family = tuple(normalized.top_k_card_ids[0]) if normalized.top_k_card_ids else ()
                if normalized.accepted and family in supported:
                    if (prediction.accepted and prediction.top_k_card_ids
                            and tuple(prediction.top_k_card_ids[0]) != family):
                        prediction = replace(prediction,accepted=False,card_id=UNKNOWN,
                            reason='hif_disabled_art_family_conflict')
                    elif not prediction.accepted:
                        prediction = normalized
                elif (selected_slot is not None and not prediction.accepted
                        and not normalized.accepted and normalized.reason == 'below_margin_threshold'):
                    from .hif_bound_title import in_candidate_band

                    weak = replace(normalized, recognizer='card_art_embedding_top5_hif_masked_v1',
                        view_mode=view.view_mode, visible_ratio=view.visible_ratio)
                    matching = [family for family in supported if in_candidate_band(weak, family,
                        min_margin if min_margin is not None else .05)]
                    if len(matching) == 1:
                        # Retain the real weak evidence for the separate
                        # same-selection continuity consumer; grant no identity.
                        prediction = normalized
                        proof.update(retained_weak_normalized=True,
                                     normalized_candidate_family=matching[0])
                elif (not prediction.accepted and not normalized.accepted
                        and family in supported and normalized.reason == 'below_acceptance_threshold'):
                    # Keep actual weak evidence for the independent selected-
                    # detail/returned-art consumer. It remains UNKNOWN here.
                    prediction = normalized
                    proof.update(retained_weak_normalized=True,
                                 normalized_candidate_family=family)
            if (not prediction.accepted and view.visible_ratio == 1.0
                    and view.canvas_offset == (0, 0) and prediction.top_k_card_ids):
                # A full card can retain distinguishing portrait features that
                # the UI-exclusion mask removes. Reuse the existing full-art
                # gate only when its independently accepted family agrees.
                normalized_full = (card.slot in corrections and proof.get('retained_weak_normalized') is True)
                tensor = preprocess_live_card_view(corrected if normalized_full else image, view,
                    source_color_order=recognizer.embedder.source_color_order)
                full = recognizer.classify_embedding(recognizer.embedder.embed_tensor(tensor),
                    card, frame_id=frame_id, plan=plan,
                    acceptance_threshold=acceptance_threshold, min_margin=min_margin,
                    resolve_upgrade_state=False)
                if normalized_full:
                    proof['normalized_full_prediction'] = full.to_dict()
                if (full.accepted and full.top_k_card_ids
                        and full.top_k_card_ids[0] == prediction.top_k_card_ids[0]):
                    predictions.append(replace(full,
                        recognizer="card_art_embedding_full_masked_agreement_v1",
                        view_mode=view.view_mode, visible_ratio=view.visible_ratio))
                    continue
            predictions.append(replace(prediction, recognizer="card_art_embedding_top5_hif_masked_v1",
                                       view_mode=view.view_mode, visible_ratio=view.visible_ratio))
            continue
        if (view.canvas_offset != (0, 0) or view.visible_ratio < MIN_SUPPORTED_VISIBLE_RATIO
                or (hif_profile and view.visible_ratio < 0.90)):
            # The legacy path admitted right-masked crops, but HIF replay
            # exposed a confident wrong family. Partial-gallery coverage is
            # unverified for this profile; do not reuse that admission here.
            predictions.append(CardPrediction(
                slot=card.slot, box=card.box, frame_id=frame_id,
                card_id=UNKNOWN, predicted_card_id=UNKNOWN, class_name=UNKNOWN,
                confidence=0.0, margin=0.0, raw_score=0.0, accepted=False,
                reason="live_art_occlusion_unverified",
                model_sha256=recognizer.model_sha256, classes_sha256=recognizer.classes_sha256,
                gallery_sha256=prepared_art.gallery_sha256 if hif_profile and prepared_art is not None
                               else recognizer.gallery_sha256,
                detector_label=card.detector_label, detector_score=card.detector_score,
                visible=card.visible, clickable=card.clickable, source_frames=(frame_id,),
                recognizer="card_art_embedding_top5_live_partial_v1",
                view_mode=view.view_mode, visible_ratio=view.visible_ratio,
            ))
            continue
        tensor = preprocess_live_card_view(
            image,
            view,
            source_color_order=recognizer.embedder.source_color_order,
        )
        prediction = recognizer.classify_embedding(
            recognizer.embedder.embed_tensor(tensor),
            card,
            frame_id=frame_id,
            plan=plan,
            acceptance_threshold=acceptance_threshold,
            min_margin=min_margin,
            resolve_upgrade_state=False,
        )
        if (hif_full_art_mask and prepared_art is not None
                and image.shape[:2] == (1280,720) and 1 <= len(cards) <= 3
                and card.box[2:] == (190,250) and view.canvas_size == (180,180)
                and view.visible_ratio == 1.0 and view.canvas_offset == (0,0)):
            # 1241/1244 have two score-preview lines over full short-hand art.
            # Reuse the already frozen HIF UI mask on this same canonical ROI.
            # A strong ordinary full view remains authoritative unless another
            # independently accepted family contradicts it; contradictions STOP.
            if prepared_art.recognizer is not recognizer:
                raise LiveCardLayoutError("HIF masked gallery belongs to a different recognizer")
            try:
                masked = prepared_art.classify(image, view, card, frame_id=frame_id, plan=plan,
                    acceptance_threshold=acceptance_threshold, min_margin=min_margin)
            except ValueError as error:
                raise LiveCardLayoutError("HIF artwork mask is not prepared") from error
            if card.slot in corrections:
                normalized_masked = prepared_art.classify(corrected,view,card,frame_id=frame_id,plan=plan,
                    acceptance_threshold=acceptance_threshold,min_margin=min_margin)
                normalized_tensor = preprocess_live_card_view(corrected,view,
                    source_color_order=recognizer.embedder.source_color_order)
                normalized_full = recognizer.classify_embedding(recognizer.embedder.embed_tensor(normalized_tensor),
                    card,frame_id=frame_id,plan=plan,acceptance_threshold=acceptance_threshold,
                    min_margin=min_margin,resolve_upgrade_state=False)
                proof = dict(corrections[card.slot],frame_id=frame_id,source_families=supported,
                    original_prediction=prediction.to_dict(),original_masked_prediction=masked.to_dict(),
                    normalized_prediction=normalized_masked.to_dict(),
                    normalized_full_prediction=normalized_full.to_dict())
                if art_preprocessing_evidence is not None:
                    art_preprocessing_evidence.append(proof)
                family = tuple(normalized_full.top_k_card_ids[0]) if normalized_full.top_k_card_ids else ()
                if (normalized_full.accepted and normalized_masked.accepted and family in supported
                        and normalized_masked.top_k_card_ids and tuple(normalized_masked.top_k_card_ids[0]) == family):
                    # A complete body and its independent 2:1 layer were
                    # proven above. Raw dark pixels are retained diagnostics,
                    # not a second valid-brightness identity observation.
                    predictions.append(replace(normalized_full,
                        recognizer='card_art_embedding_full_masked_agreement_v1',
                        view_mode=view.view_mode,visible_ratio=view.visible_ratio))
                else:
                    # Never fall back to a confident uncorrected dark crop.
                    predictions.append(replace(prediction,accepted=False,card_id=UNKNOWN,
                        reason='hif_disabled_art_normalized_unresolved'))
                continue
            if masked.accepted and masked.top_k_card_ids:
                # Legacy observation may use .70, whereas the frozen HIF
                # mask enforces .85/.05. A weak legacy family cannot veto an
                # independently accepted current masked identity.
                full_identity_qualified = (prediction.accepted
                    and prediction.confidence >= max(.85, acceptance_threshold)
                    and prediction.margin >= max(.05, min_margin))
                if (full_identity_qualified and prediction.top_k_card_ids
                        and prediction.top_k_card_ids[0] != masked.top_k_card_ids[0]):
                    predictions.append(replace(prediction, accepted=False, card_id=UNKNOWN,
                        reason='hif_full_masked_family_conflict',
                        recognizer='card_art_embedding_full_masked_conflict_v1',
                        view_mode=view.view_mode, visible_ratio=view.visible_ratio))
                    continue
                if not full_identity_qualified:
                    if art_preprocessing_evidence is not None:
                        art_preprocessing_evidence.append(dict(
                            scope='hif_current_mask_over_weak_full', frame_id=frame_id, slot=card.slot,
                            original_prediction=prediction.to_dict(), masked_prediction=masked.to_dict()))
                    predictions.append(replace(masked,
                        recognizer='card_art_embedding_top5_hif_masked_v1',
                        view_mode=view.view_mode, visible_ratio=view.visible_ratio))
                    continue
        if (refine_hif_title_anchor and not prediction.accepted
                and prediction.reason == 'below_acceptance_threshold'
                and prediction.top_k_card_ids and len(cards) in (2,3)
                and image.shape[:2] == (1280,720) and view.visible_ratio == 1.0
                and view.canvas_offset == (0,0) and card.box[2:] == (190,250)):
            from .hif_visual import _hif_unselected_title_plates
            plates = _hif_unselected_title_plates(image)
            if len(plates) == len(cards):
                px,py,pw,ph = plates[card.slot]
                # The full title interior is 185 px. One antialiased column
                # can extend its left edge to 186 px while the body stays put.
                # Center that independently measured interior; do not search
                # multiple crops for the highest identity score.
                if (pw == 186 and px-2 == card.box[0]
                        and py+ph+2-250 == card.box[1]):
                    x,y,w,h = view.visible_box
                    centered = replace(view, visible_box=(x+1,y,w,h))
                    tensor = preprocess_live_card_view(image, centered,
                        source_color_order=recognizer.embedder.source_color_order)
                    corrected = recognizer.classify_embedding(
                        recognizer.embedder.embed_tensor(tensor), card, frame_id=frame_id, plan=plan,
                        acceptance_threshold=acceptance_threshold, min_margin=min_margin,
                        resolve_upgrade_state=False)
                    if (corrected.accepted and corrected.top_k_card_ids
                            and corrected.top_k_card_ids[0] == prediction.top_k_card_ids[0]):
                        predictions.append(replace(corrected,
                            recognizer='card_art_embedding_title_center_v1',
                            view_mode=view.view_mode, visible_ratio=view.visible_ratio))
                        continue
        predictions.append(
            replace(
                prediction,
                recognizer="card_art_embedding_top5_live_partial_v1",
                view_mode=view.view_mode,
                visible_ratio=view.visible_ratio,
            )
        )
    return tuple(predictions)
