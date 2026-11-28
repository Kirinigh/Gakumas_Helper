import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("classification", ROOT / "tools/card-embedding/build_skill_card_classification.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def example(source="produce", rarity="N", name="カード", **origins):
    return (
        dict(id=1, plan="free", rarity=rarity, type="active", sourceType=source, name=name),
        dict(planType="ProducePlanType_Common", rarity="ProduceCardRarity_" + rarity, **origins),
    )


@pytest.mark.parametrize("source", ["produce", "pIdol", "support", "default"])
def test_legend_always_other(source):
    card, master = example(source, "L", "基本カード", originPrimaStellaIdolCardId="idol")
    master["rarity"] = "ProduceCardRarity_Legend"
    assert BUILDER.classify(card, master)[0]["kind"] == "other"


@pytest.mark.parametrize(
    "source,name,kind",
    [
        ("pIdol", "カード", "p_idol"),
        ("support", "カード", "support"),
        ("default", "基本カード", "basic"),
        ("produce", "カード", "other"),
    ],
)
def test_explicit_catalog_source_without_official_owner(source, name, kind):
    assert BUILDER.classify(*example(source, name=name))[0]["kind"] == kind


def test_trouble_uses_official_n():
    card, master = example()
    card.update(rarity="T", type="trouble")
    assert BUILDER.classify(card, master)[0]["rarity"] == "N"
    card["type"] = "active"
    with pytest.raises(ValueError, match="rarity conflict"):
        BUILDER.classify(card, master)


def test_conflicting_source_rejected():
    with pytest.raises(ValueError, match="conflicting ownership"):
        BUILDER.classify(*example("pIdol", originSupportCardId="support"))
    with pytest.raises(ValueError, match="unknown card source"):
        BUILDER.classify(*example("unknown"))
    card, master = example()
    card["plan"] = "sense"
    with pytest.raises(ValueError, match="plan conflict"):
        BUILDER.classify(card, master)


def test_complete_artifact_and_partitions():
    index = BUILDER.read(ROOT / "assets/data/skill_card_classification.json")
    cards = {c["business_id"]: c for c in index["cards"]}
    assert list(cards) == list(range(1, 936))
    for facet, groups in index["indexes"].items():
        assert sorted(i for ids in groups.values() for i in ids) == list(cards)
        for value, ids in groups.items():
            assert ids == [i for i, c in cards.items() if c[facet] == value]
            assert index["counts"][facet][value] == len(ids)
    assert len(index["indexes"]["rarity"]["L"]) == 28
    assert all(cards[i]["kind"] == "other" for i in index["indexes"]["rarity"]["L"])
    assert cards[875]["kind"] == cards[876]["kind"] == "other"
    assert cards[23]["rarity"] == "N" and cards[23]["source"]["catalog_rarity"] == "T"
    assert cards[757]["kind"] == cards[832]["kind"] == "p_idol"
    pairs = {}
    for card in cards.values():
        labels = tuple(card[f] for f in BUILDER.FACETS)
        assert pairs.setdefault(card["internal_id"], labels) == labels
    assert not index["runtime_filtering_enabled"]


def test_filters_union_intersection_and_invalid_values():
    index = BUILDER.read(ROOT / "assets/data/skill_card_classification.json")
    expected = [c["business_id"] for c in index["cards"] if c["plan"] in {"free", "sense"} and c["rarity"] == "L"]
    assert BUILDER.select_ids(index, plan=["free", "sense"], rarity="L") == expected
    assert BUILDER.select_ids(index) == list(range(1, 936))
    assert BUILDER.select_ids(index, plan=[]) == []
    for filters in [dict(plan="unknown"), dict(unknown="sense")]:
        with pytest.raises(ValueError):
            BUILDER.select_ids(index, **filters)


def test_duplicate_source_identity_rejected():
    with pytest.raises(ValueError, match="duplicate identity"):
        BUILDER.indexed([{"id": 1}, {"id": 1}], lambda r: r["id"])
