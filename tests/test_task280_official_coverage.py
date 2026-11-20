import pytest

from tools.task280_shared_rules import SUMMER, SUMMER_FORM, SUMMER_EFFECTS, shared_card_candidates
from tools.task280_official_coverage import _gap, card_form, verified_produce_only_items


def test_compact_and_expanded_forms_preserve_official_upgrade_and_customization():
    compact = 'p_card-01-men-3_034+2/p_card_custom-wrapper-p_card-01-men-3_034-a-01*2'
    expanded = {
        'produce_card_id': 'p_card-01-men-3_034', 'current_upgrade_count': 2,
        'customizes': [{'id': 'p_card_custom-wrapper-p_card-01-men-3_034-a-01', 'customize_count': 2}],
    }
    assert card_form(compact) == card_form(expanded) == ('p_card-01-men-3_034', 2, True)
    assert card_form('p_card-01-men-3_034') == ('p_card-01-men-3_034', 0, False)
    with pytest.raises(ValueError, match='invalid_compact'):
        card_form('p_card-01-men-3_034++')


def test_missing_identity_is_distinct_from_uncompiled_effect():
    assert _gap('x', {}, {}, {}) == 'master_missing'
    assert _gap('x', {'x': 'name'}, {'name': ['1', '2']}, {'1': None}) == 'local_identity_unresolved'
    assert _gap('x', {'x': 'name'}, {'name': ['1']}, {}) == 'local_effect_not_compiled'
    assert _gap('x', {'x': 'name'}, {'name': ['1']}, {'1': None}) is None


def test_shared_candidate_is_tied_to_verified_ordinary_form_and_effects():
    raw_id, upgrade = SUMMER_FORM
    master = dict(id=raw_id, upgradeCount=upgrade, name='夏夜に咲く思い出+',
                  playEffects=[{'produceExamEffectId': key} for key in SUMMER_EFFECTS])
    local = {'id': SUMMER, 'name': master['name']}
    assert shared_card_candidates([master], [local])[SUMMER_FORM].card_id == SUMMER
    assert shared_card_candidates([master | {'upgradeCount': 2}], [local]) == {}
    assert shared_card_candidates([master | {'playEffects': master['playEffects'][:2]}], [local]) == {}


def test_source_item_scope_needs_unique_matching_produce_only_semantics():
    official = {'pitem-original': {'name': 'same title', 'rarity': 'SSR'}}
    ris = {'name': 'same title', 'rarity': 'SSR', 'mode': 'produce', 'effects': ''}
    assert verified_produce_only_items(official, [ris]) == {'pitem-original'}
    assert not verified_produce_only_items(official, [ris | {'mode': 'stage'}])
    assert not verified_produce_only_items(official, [ris, ris | {'mode': 'stage'}])
    assert not verified_produce_only_items(official | {'duplicate': official['pitem-original']}, [ris])
