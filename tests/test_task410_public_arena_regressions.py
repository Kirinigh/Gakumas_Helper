"""Synthetic public regressions for arena task ownership and evidence."""

import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from agent.arena_winrate.challenge_flow import ArenaChallengeFlowError, ArenaChallengeRecordStore
from agent.arena_winrate.badge_reference import CardIdentityCache, BadgeReferenceError, BadgeReferenceGallery
from agent.arena_winrate._arena_page_flow import _has_first_opponent_stage_total


def test_new_task_retires_only_an_inactive_pending_marker(tmp_path):
    store = ArenaChallengeRecordStore(tmp_path)
    store.begin({"capture_id": "previous", "origin_run_id": "old"})
    pending = store.pending_path.read_bytes()
    assert store.retire_previous_task("old") is None
    with pytest.raises(ArenaChallengeFlowError, match="another active task"):
        store.retire_previous_task("new", active_run_ids=("old",))
    assert store.pending_path.read_bytes() == pending

    store.results_root.mkdir()
    archived = store.results_root / "previous.json"
    archived.write_text('{"synthetic":true}', encoding="utf-8")
    assert store.retire_previous_task("new") == "previous"
    assert store.load_pending() is None
    assert archived.read_text(encoding="utf-8") == '{"synthetic":true}'


def test_existing_challenge_entry_retains_result_wait_and_communication_guard():
    root = Path(__file__).resolve().parents[1]
    pipeline = json.loads((root / "assets/resource/base/pipeline/Challenge.json").read_text(encoding="utf-8"))
    assert pipeline["Challenge"]["next"][0] == "ChallengeResetOwnScorePreparation"
    assert pipeline["ArenaChallengeAwaitFormalResult"]["next"] == [
        "ArenaChallengeFormalResultPage", "ArenaChallengeResultCommunication",
    ]
    assert pipeline["ArenaChallengeResultCommunication"]["action"] == {"type": "DoNothing"}
    assert "ArenaChallengeResumeBattle" not in pipeline


def test_first_opponent_total_requires_unique_same_frame_numeric_evidence():
    def atom(text, x, y, width=74, height=30):
        return SimpleNamespace(text=text, box=(x, y, width, height))

    frame = SimpleNamespace(shape=(1280, 720, 3))
    exact_totals = [atom("総合力", x, y) for y in (440, 610, 780) for x in (60, 414)
                    if (x, y) != (414, 440)]
    candidate = atom("截合力", 414, 440)
    numbers = [atom(str(50000 + index), x, y, width=76)
               for index, (x, y) in enumerate((x, y) for y in (440, 610, 780) for x in (148, 504))]
    surrounding = [atom(f"ステージ{index}", x, 320, width=120)
                   for index, x in enumerate((48, 298, 548), 1)]
    surrounding.extend((atom("サポートボーナス", 42, 1080, width=150), atom("挑戦開始", 285, 1080, width=150)))
    items = [*exact_totals, candidate, *numbers, *surrounding]
    assert _has_first_opponent_stage_total(frame, items, exact_totals)
    assert not _has_first_opponent_stage_total(frame, [*items, atom("99999", 512, 440)], exact_totals)
    assert not _has_first_opponent_stage_total(frame, items[:-1], exact_totals)


def test_ranked_identity_keeps_inactive_rows_out_and_preserves_tie_order(monkeypatch):
    gallery = BadgeReferenceGallery(
        business_ids=(999, 1, 2, 3, 4),
        visual_group_ids=("placeholder", "z", "a", "a", "x"),
        reference_names=("inactive", "first", "second", "third", "fourth"),
        top_coarse=np.stack([np.full((8, 16, 3), value, dtype=np.uint8) for value in (0, 2, 2, 2, 5)]),
        maximum_coarse_error=10,
        minimum_group_margin=1.0,
        provisional_shared_pairs=((2, 3),),
        reference_enabled=np.array([False, True, True, True, True]),
    )
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    box = (0, 0, 96, 96)
    resolver = Mock(wraps=gallery._resolve_identity_query)
    monkeypatch.setattr(gallery, "_resolve_identity_query", resolver)
    cache = CardIdentityCache()

    group, signature, identity = gallery.content_signature_with_identity(frame, box, identity_cache=cache)
    assert group == "z"
    assert identity["top_group"] == "a"
    assert [item["reference_business_id"] for item in identity["candidates"]] == [2, 3, 1]
    assert identity == gallery.measure_identity(frame, box)
    identity["candidates"][0]["reference_business_id"] = 999
    frame[48:] = 5
    _, fresh_signature, cached = gallery.content_signature_with_identity(frame, box, identity_cache=cache)
    resolver.assert_called_once()
    assert cached["candidates"][0]["reference_business_id"] == 2
    assert not np.array_equal(signature, fresh_signature)

    selected = frozenset({999, 4})
    group, _, scoped = gallery.content_signature_with_identity(frame, box, eligible_card_ids=selected,
                                                              identity_cache=cache)
    assert (group, scoped["reference_business_id"]) == ("x", 4)
    assert scoped == gallery.measure_identity(frame, box, eligible_card_ids=selected)
    frame[:48] = 2
    group, _, changed = gallery.content_signature_with_identity(frame, box, eligible_card_ids=selected,
                                                                identity_cache=cache)
    assert (group, changed["reference_business_id"], changed["identity_top_error"]) == ("x", 4, 3.0)
    assert resolver.call_count == 3
    with pytest.raises(BadgeReferenceError, match="no clean-card references"):
        gallery.content_signature_with_identity(frame, box, eligible_card_ids=frozenset({999}), identity_cache=cache)
