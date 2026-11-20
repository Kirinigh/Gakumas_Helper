"""Offline regressions for hand, P-item and drink observations using synthetic frames."""

from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from agent.card_selection.types import CandidateBox, CardPrediction
from agent.live_decision.catalog import SkillCardEntry, SkillCardFeatureCatalog
from agent.live_decision.hif_visual import hif_select_label_rois, hif_current_title_family, locate_hif_selected_detail_title
from agent.live_decision.item_panel import read_item_panel, item_counter_consensus, read_explicit_item_counter
from agent.live_decision.drink_icons import EXAM_CENTERS, DrinkIconRecognizer
from agent.live_decision.drink_panel import DrinkDetailEvidence, observe_drink_inventory


def test_selected_title_rejects_two_panels_and_rois_reject_shifted_hand():
    frame = np.zeros((1280, 720, 3), np.uint8)
    frame[500:820, 95:625] = 255
    assert locate_hif_selected_detail_title(frame) == (111, 506, 446, 44)
    frame[300:420, 95:625] = 255
    assert locate_hif_selected_detail_title(frame) is None

    hand = tuple(CandidateBox(i, (x, 884, 190, 250)) for i, x in enumerate((48, 265, 482)))
    assert len(hif_select_label_rois(frame.shape, hand)) == 3
    shifted = (replace(hand[0], box=(72, 884, 190, 250)), *hand[1:])
    with pytest.raises(ValueError, match="anchors"):
        hif_select_label_rois(frame.shape, shifted)


@pytest.mark.parametrize("fault", [None, "stale", "truncated", "ambiguous", "conflict"])
def test_weak_card_image_uses_only_current_complete_compatible_title(fault):
    frame = np.zeros((1280, 720, 3), np.uint8)
    hand = tuple(CandidateBox(i, (x, 884, 190, 250)) for i, x in enumerate((48, 265, 482)))
    prediction = CardPrediction(
        1,
        hand[1].box,
        "current",
        "UNKNOWN",
        "1",
        "Plausible",
        0.86,
        0.02,
        0.86,
        False,
        "below_margin_threshold",
        "model",
        "classes",
        source_frames=("current",),
        recognizer="card_art_embedding_top5_live_partial_v1",
        top_k_card_ids=(("1",), ("2", "3")),
        top_k_scores=(0.86, 0.84),
    )
    names = {"1": "Aria", "2": "Focus", "3": "Focus+", "4": "Another"}
    catalog = SkillCardFeatureCatalog({key: SkillCardEntry(key, name, "skill", "free", key == "3", 0, "", "") for key, name in names.items()})
    title = {"truncated": "Focu", "conflict": "Another"}.get(fault, "Focus")
    read_ocr = Mock(return_value=[SimpleNamespace(box=(3, 4, 71, 20), text=title, score=0.98)])
    plates = ((267, 1105, 180, 25),) * (2 if fault == "ambiguous" else 1)
    evidence, _ = hif_current_title_family(
        read_ocr,
        frame,
        hand,
        prediction,
        frame_id="older" if fault == "stale" else "current",
        plan="sense",
        minimum_margin=0.05,
        plates=plates,
        catalog=catalog,
    )
    if fault is None:
        assert evidence["accepted"] and evidence["family"] == ("2", "3")
        assert evidence["upgrade_verified"] is False
    elif fault == "conflict":
        assert evidence["accepted"] is False
    else:
        assert evidence is None
    if fault in {"stale", "ambiguous"}:
        read_ocr.assert_not_called()


def test_explicit_item_badge_zero_requires_valid_read_and_agreement():
    image = np.zeros((50, 50, 3), np.uint8)
    cv2.circle(image, (25, 25), 15, (135, 135, 135), -1)
    cv2.putText(image, "0", (19, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    read_ocr = Mock(return_value=[SimpleNamespace(box=(4, 4, 11, 11), text="０", score=0.987)])
    zero = read_explicit_item_counter(read_ocr, image, (0, 0, 50, 50), use_limit=3)
    assert item_counter_consensus((zero, zero, zero)).value == 0
    with pytest.raises(ValueError, match="consensus"):
        item_counter_consensus((zero, zero, replace(zero, value=1)))
    with pytest.raises(ValueError, match="consensus"):
        item_counter_consensus((zero,))
    with pytest.raises(ValueError, match="missing or ambiguous"):
        read_explicit_item_counter(read_ocr, np.zeros_like(image), (0, 0, 50, 50), use_limit=3)
    assert read_ocr.call_count == 1


def test_item_overview_retains_missing_counter_and_rejects_duplicate_name():
    image = np.zeros((1280, 720, 3), np.uint8)
    image[125:305, 24:697] = 255
    image[117:992, 703:705] = 132
    markers = [
        SimpleNamespace(box=(22, 34, 110, 20), text="所持Pアイテム", score=0.97),
        SimpleNamespace(box=(230, 1127, 81, 20), text="閉じる", score=0.97),
    ]
    item = SimpleNamespace(box=(240, 132, 140, 25), text="検証用アイテム", score=0.98)
    observed = read_item_panel(lambda *_: (*markers, item), image)
    assert observed.rows[0].name == item.text
    assert observed.rows[0].used_count is None and observed.rows[0].use_limit is None
    with pytest.raises(ValueError, match="name is missing or ambiguous"):
        read_item_panel(lambda *_: (*markers, item, item), image)


def _drink_frames():
    frames = tuple(np.zeros((1280, 720, 3), np.uint8) for _ in range(3))
    rng = np.random.default_rng(90)
    icons = {key: rng.integers(0, 256, (56, 56, 4), dtype=np.uint8) for key in ("a", "b")}
    for icon in icons.values():
        icon[..., 3] = 255
    x, y = EXAM_CENTERS[0]
    for frame in frames:
        frame[y - 28 : y + 28, x - 28 : x + 28] = icons["a"][..., :3]
    return frames, icons


def test_drink_template_only_resolves_with_complete_catalog_and_three_frames():
    frames, icons = _drink_frames()
    accepted = DrinkIconRecognizer(icons, catalog_ids=icons)(frames, (0,), client_size=(720, 1280))
    assert accepted[0]["drink_id"] == "a"
    incomplete = DrinkIconRecognizer(icons, catalog_ids=(*icons, "c"))(frames, (0,), client_size=(720, 1280))
    assert incomplete[0]["drink_id"] is None
    x, y = EXAM_CENTERS[0]
    frames[-1][y - 28 : y + 28, x - 28 : x + 28] = icons["b"][..., :3]
    unstable = DrinkIconRecognizer(icons, catalog_ids=icons)(frames, (0,), client_size=(720, 1280))
    assert unstable[0]["drink_id"] is None


@pytest.mark.parametrize("fault", [None, "no_scope", "empty_slot", "conflict", "wrong_slot", "stale", "changed"])
def test_drink_detail_only_accepts_authorized_slot_and_independent_unchanged_return(fault):
    frames, _ = _drink_frames()
    ids = ("a", "b", "c")
    occupied = ((27, 1165, 70, 70), None, None, None)
    empty = ((27, 1165, 36, 45) if fault == "conflict" else None, (129, 1177, 36, 45), (214, 1177, 36, 45), (300, 1177, 36, 45))
    callback = Mock(
        return_value=DrinkDetailEvidence(
            1 if fault == "wrong_slot" else 0,
            "a",
            0.99,
            ids,
            ids if fault == "stale" else ("d", "e", "f"),
            fault != "changed",
        )
    )
    result = observe_drink_inventory(
        frames,
        (occupied,) * 3,
        (empty,) * 3,
        client_size=(720, 1280),
        frame_ids=ids,
        catalog_ids=("a", "b"),
        authorized_slot={"no_scope": None, "empty_slot": 1}.get(fault, 0),
        read_detail=callback,
    )
    assert result[0].drink_id == ("a" if fault is None else None)
    assert result[0].state == ("UNKNOWN" if fault == "conflict" else "OCCUPIED")
    assert all(slot.state == "EMPTY" for slot in result[1:])
    if fault in {"no_scope", "empty_slot", "conflict"}:
        callback.assert_not_called()
    else:
        callback.assert_called_once_with(slot=0, frame_ids=ids)
