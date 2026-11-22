"""Offline regressions for hand, P-item and drink observations using synthetic frames."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from agent.card_selection.types import CandidateBox, CardPrediction
from agent.live_decision import drink_panel
from agent.live_decision.catalog import SkillCardEntry, SkillCardFeatureCatalog
from agent.live_decision.hif_visual import hif_select_label_rois, hif_current_title_family, locate_hif_selected_detail_title
from agent.live_decision.item_panel import read_item_panel, item_counter_consensus, read_explicit_item_counter
from agent.live_decision.drink_icons import EXAM_CENTERS, DrinkIconRecognizer
from agent.live_decision.status_panel import StatusInspectionError


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


def _ocr_context(rows):
    return SimpleNamespace(run_recognition=Mock(return_value=SimpleNamespace(hit=True, filtered_results=rows)))


def test_explicit_item_badge_zero_requires_valid_read_and_agreement():
    image = np.zeros((50, 50, 3), np.uint8)
    cv2.circle(image, (25, 25), 15, (135, 135, 135), -1)
    cv2.putText(image, "0", (19, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    context = _ocr_context([SimpleNamespace(box=(4, 4, 11, 11), text="０", score=0.987)])
    zero = read_explicit_item_counter(context, image, (0, 0, 50, 50), use_limit=3)
    assert item_counter_consensus((zero,)).value == 0
    with pytest.raises(ValueError, match="consensus"):
        item_counter_consensus((zero, replace(zero, value=1)))
    with pytest.raises(ValueError, match="consensus"):
        item_counter_consensus(())
    with pytest.raises(ValueError, match="invalid"):
        item_counter_consensus((replace(zero, confidence=0.90),))
    with pytest.raises(ValueError, match="missing or ambiguous"):
        read_explicit_item_counter(context, np.zeros_like(image), (0, 0, 50, 50), use_limit=3)
    assert context.run_recognition.call_count == 1


def test_item_overview_retains_missing_counter_and_rejects_duplicate_name():
    image = np.zeros((1280, 720, 3), np.uint8)
    image[125:305, 24:697] = 255
    image[117:992, 703:705] = 132
    markers = [
        SimpleNamespace(box=(22, 34, 110, 20), text="所持Pアイテム", score=0.97),
        SimpleNamespace(box=(230, 1127, 81, 20), text="閉じる", score=0.97),
    ]
    item = SimpleNamespace(box=(240, 132, 140, 25), text="検証用アイテム", score=0.98)
    observed = read_item_panel(_ocr_context((*markers, item)), image)
    assert observed.rows[0].name == item.text
    assert observed.rows[0].used_count is None and observed.rows[0].use_limit is None
    with pytest.raises(ValueError, match="name is missing or ambiguous"):
        read_item_panel(_ocr_context((*markers, item, item)), image)


def test_drink_template_resolves_one_current_frame_with_complete_catalog():
    frame = np.zeros((1280, 720, 3), np.uint8)
    rng = np.random.default_rng(90)
    icons = {key: rng.integers(0, 256, (56, 56, 4), dtype=np.uint8) for key in ("a", "b")}
    for icon in icons.values():
        icon[..., 3] = 255
    x, y = EXAM_CENTERS[0]
    crop = frame[y - 28 : y + 28, x - 28 : x + 28]
    crop[:] = icons["a"][..., :3]
    recognizer = DrinkIconRecognizer(icons, catalog_ids=icons)
    accepted = recognizer.read_slots(frame, client_size=(720, 1280))
    assert accepted[0]["drink_id"] == "a"
    incomplete = DrinkIconRecognizer(icons, catalog_ids=(*icons, "c")).read_slots(frame, client_size=(720, 1280))
    assert incomplete[0]["reason"] == "incomplete_icon_catalog"
    assert incomplete[0]["drink_id"] is None
    crop[:] = cv2.addWeighted(icons["a"][..., :3], 0.5, icons["b"][..., :3], 0.5, 0)
    weak = recognizer.read_slots(frame, client_size=(720, 1280))
    assert weak[0]["reason"] == "weak_icon"
    assert weak[0]["drink_id"] is None


@pytest.mark.parametrize("fault", [None, "no_scope", "empty_slot", "conflict", "denied", "stale", "changed", "low_confidence"])
def test_drink_detail_requires_authorized_open_cancel_and_unchanged_return(fault, monkeypatch):
    frames = {name: np.zeros((1280, 720, 3), np.uint8) for name in ("hud", "before", "detail", "cancel", "after")}
    frame_ids = {id(frame): name for name, frame in frames.items()}
    capture = iter(frames.values())

    def frame_id(frame):
        return frame_ids[id(frame)]

    def post_screencap():
        frame = next(capture)
        return SimpleNamespace(wait=lambda: SimpleNamespace(get=lambda: frame))

    def checkpoint(evidence, *, single_frame):
        assert single_frame is True
        frame, = evidence
        return dict(families=(), fields={}, frame_ids=(frame_id(frame),))

    occupied = ((27, 1165, 70, 70), None, None, None)
    shifted = ((30, 1165, 70, 70), None, None, None)
    empty = tuple(dict(box=(x, 1177, 36, 45), confidence=1.0) for x in (44, 129, 214, 300))
    monkeypatch.setattr(drink_panel, "drink_slot_targets", lambda _, frame:
                        shifted if fault == "changed" and frame is frames["after"] else occupied)
    monkeypatch.setattr(drink_panel, "empty_drink_slot_targets", lambda *_: empty if fault == "conflict" else (None, *empty[1:]))
    monkeypatch.setattr(drink_panel, "read_drink_detail", lambda *_, **__: dict(
        name="A", name_ink_box=(90, 330, 100, 30), cancel_box=(20, 980, 140, 40), available=True,
        confidence=0.90 if fault == "low_confidence" else 0.99))
    if fault == "stale":
        monkeypatch.setattr(drink_panel, "frame_captured_at", lambda *_: drink_panel.time.perf_counter() - 2)
    monkeypatch.setattr("utils.maa_input.verify_maa_input_endpoints", lambda *_, **__: {})
    direct_click = Mock(side_effect=AssertionError("direct input instead of scoped navigation"))
    controller = SimpleNamespace(info={}, post_click=direct_click, post_screencap=post_screencap,
                                 set_screenshot_target_long_side=lambda size: size == 1280)
    context = SimpleNamespace(tasker=SimpleNamespace(controller=controller))
    navigate = Mock(return_value=fault != "denied")

    def observe():
        return drink_panel.observe_complete_drink_inventory(
            context, capture_frames=lambda: (next(capture),), checkpoint=checkpoint, frame_identifier=frame_id,
            metadata=(dict(id="a", name="A", plan="free"),), template=None, plan="sense", run_id="fixture", revision=4,
            single_frame_hud=True, hud_detail_slot={"no_scope": None, "empty_slot": 1}.get(fault, 0),
            hud_detail_navigate=None if fault == "no_scope" else navigate,
        )

    if fault is None:
        result, = observe()
        assert result["full_drink_state_ready"] is True
        assert result["returned_unchanged"] is True
        assert result["drinks"] == ("a",)
        assert result["frame_ids"] == ("after",)
        assert result["slots"][0]["source"] == "scoped_opening_detail"
        assert [slot["state"] for slot in result["slots"][1:]] == ["EMPTY"] * 3
    else:
        with pytest.raises(StatusInspectionError) as error:
            observe()
        result, = error.value.events
        assert result["accepted"] is False
        assert result["slots"][0]["drink_id"] is None
        assert result["slots"][0]["state"] == ("UNKNOWN" if fault == "conflict" else "OCCUPIED")

    phases = [call.kwargs["phase"] for call in navigate.call_args_list]
    if fault in {"no_scope", "empty_slot", "conflict", "stale"}:
        assert phases == []
    elif fault == "denied":
        assert phases == ["open"]
    else:
        assert phases == ["open", "cancel"]
    assert all(call.kwargs["slot"] == 0 and call.kwargs["run_id"] == "fixture" and call.kwargs["revision"] == 4
               for call in navigate.call_args_list)
    direct_click.assert_not_called()
