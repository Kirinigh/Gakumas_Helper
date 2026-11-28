from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from agent.live_decision import idol_road_formation as formation
from agent.live_decision.idol_road_formation import (
    FormationThumbnailMatcher,
    find_idol_card,
    read_formation_deck,
    resolve_formation_idol,
    read_formation_overview,
)
from agent.live_decision.high_score.contracts import PlanType


def text(value, x, y):
    return dict(text=value, box=(x - 10, y - 8, 20, 16))


def test_reads_hiro_formation_fields_in_screen_order():
    rows = [text('STAGE14', 334, 92), text('ターン', 115, 95), text('12', 115, 56),
            text('一匹狼', 260, 370), text('月村手毬', 269, 405), text('体力', 333, 539),
            text('38', 333, 575), text('1/2', 365, 658), text('挑戦する', 357, 1141),
            text('2221％', 643, 551), text('1304%', 643, 510), text('1800%', 643, 530)]
    page = read_formation_overview(rows)
    assert (page.page, page.layer, page.turns, page.max_stamina) == (1, 14, 12, 38)
    assert (page.idol_name, page.card_title) == ('月村手毬', '一匹狼')
    assert page.multiplier_permils == (13040, 18000, 22210)
    assert page.next_page == (462, 658)
    assert read_formation_overview(rows[:-1]).multiplier_permils is None
    assert read_formation_overview([rows[0], rows[8]]) is None
    deck = read_formation_overview([rows[0], text('2/2', 365, 658), rows[8]])
    assert deck.page == 2 and deck.previous_page == (268, 658)
    assert deck.turns is None and deck.multiplier_permils is None


@pytest.mark.parametrize(('support', 'pager'), [('応援/トラブル', '1/2'), ('応援／トラプル', '1／2')])
def test_reads_sena_floor_when_challenge_text_is_obscured(support, pager):
    rows = [text('STAGE 12', 83, 56), text(support, 444, 25),
            text('審查基準', 260, 312), text('ターン', 295, 91), text('20', 296, 54),
            text('ガラクタロード', 264, 371), text('十王星南', 263, 403),
            text('体力', 530, 547), text('57', 529, 576), text(pager, 359, 672),
            text('挑戦す', 363, 1089), text('1666%', 641, 395),
            text('1071%', 639, 456), text('2185%', 642, 514)]
    page = read_formation_overview(rows)
    assert (page.page, page.layer, page.turns, page.max_stamina) == (1, 12, 20, 57)
    assert (page.idol_name, page.card_title) == ('十王星南', 'ガラクタロード')
    assert page.multiplier_permils == (16660, 10710, 21850)
    assert page.next_page == (456, 672)
    assert read_formation_overview(rows + [text('おすすめ効果', 132, 956),
        text('好印象', 156, 993)]).recommended_plan is PlanType.LOGIC
    assert read_formation_overview(rows + [text('好印象', 156, 993)]).recommended_plan is None


def test_formation_page_requires_a_centered_pager_and_other_page_evidence():
    rows = [text('STAGE 21', 83, 56), text('応援/トラプル', 444, 25), text('ターン', 295, 91)]
    assert read_formation_overview([*rows, text('1/2', 359, 672)]) is None
    assert read_formation_overview([*rows, text('1/2', 570, 672), text('挑戦する', 360, 1089)]) is None


def test_keeps_duplicate_and_unread_active_cards_without_counting_disabled_slots():
    image = np.zeros((1280, 720, 3), dtype=np.uint8)
    image[373:457, 28:112] = 205
    image[373:457, 118:202] = 205
    image[373:457, 208:292] = 205
    image[373:457, 298:382] = 130  # disabled by the game
    image[473:557, 153:237] = 190  # active but unread
    for x, y in ((29, 373), (120, 373), (211, 373), (154, 473)):
        image[y + 22:y + 52, x + 20:x + 50] = (205, 175, 140)
    image[395:401, 178:185] = (0, 0, 240)  # the second card's red plus
    calls = []

    def identify(frame, box, upgraded, column):
        assert frame is image
        calls.append((box, upgraded))
        return {29: '7', 120: '8', 211: '7'}.get(box[0])

    cards = read_formation_deck(image, identify)
    assert [(c.row, c.column, c.card_id, c.upgraded) for c in cards] == [
        (0, 0, '7', False), (0, 1, '8', True), (0, 2, '7', False), (1, 0, None, False),
    ]
    assert len(calls) == 4


def test_four_memory_top_rows_follow_their_borders_during_scroll_rebound():
    image = np.full((1280, 720, 3), 158, dtype=np.uint8)
    for top, start in ((396, 29), (492, 154)):
        for column in range(6):
            x = start + 91 * column
            image[top - 2:top + 1, x:x + 82] = 70
            image[top + 1:top + 83, x:x + 82] = 205
            image[top + 22:top + 52, x + 20:x + 50] = (205, 175, 140)
    cards = read_formation_deck(image, lambda _frame, box, *_: str(box[1]), with_fourth=True)
    assert [(card.row, card.column, card.card_id) for card in cards] == [
        (row, column, top) for row, top in enumerate(('396', '492')) for column in range(6)
    ]


@pytest.mark.parametrize('shift', [0, 15])
def test_scrolled_formation_uses_the_full_row_above_sparse_gray_placeholders(shift):
    image = np.full((1280, 720, 3), 246, dtype=np.uint8)
    for top, count in ((460 + shift, 6), (556 + shift, 2)):
        for column in range(6):
            x = 154 + 91 * column
            if column >= count:
                image[top:top + 84, x:x + 82] = 222
                image[top + 42:top + 84, x:x + 82] = 229
                continue
            image[top - 2:top + 1, x:x + 82] = 70
            image[top + 1:top + 80, x:x + 82] = 130 if column == 0 else 225
            image[top + 20:top + 65, x + 12:x + 56] = (140, 164, 196) if column else (82, 100, 130)
            image[top + 80:top + 83, x:x + 82] = 70

    crops = []

    def identify(_image, box, _upgraded, column):
        crops.append((box[1], column))
        return str(column) if column != 5 else None

    cards = read_formation_deck(image, identify, after_scroll=True)
    assert crops == [(460 + shift, column) for column in range(1, 6)] + [(556 + shift, 1)]
    assert [(card.row, card.column, card.card_id) for card in cards] == [
        (2, 1, '1'), (2, 2, '2'), (2, 3, '3'), (2, 4, '4'), (2, 5, None), (3, 1, '1'),
    ]


def test_tower_matches_arena_slots_and_uses_nearest_neighbor_to_disambiguate():
    rows = [
        (871, 'logic', 'p_idol', False, 'sena'),
        (872, 'logic', 'p_idol', True, 'sena'),
        (874, 'logic', 'p_idol', True, 'other-idol'),
        (99, 'logic', 'support', True, 'support'),
        (103, 'free', 'other', True, 'common'),
        (105, 'sense', 'p_idol', True, 'sense'),
    ]
    scopes = []
    fallback = []

    def reference(_image, _box, *, eligible_card_ids):
        scopes.append(eligible_card_ids)
        groups = {group for cid, _, _, _, group in rows if cid in eligible_card_ids}
        return dict(candidates=[dict(status='MEASURED', reference_visual_group=group,
            identity_low_confidence=False) for group in groups])

    def near(_vector, *, eligible_card_ids):
        fallback.append(eligible_card_ids)
        group = 'sena' if 872 in eligible_card_ids else 'support'
        return [SimpleNamespace(visual_group_id=group)]

    matcher = FormationThumbnailMatcher(
        SimpleNamespace(business_ids=[row[0] for row in rows],
                        visual_group_ids=[row[-1] for row in rows], measure_identity=reference),
        SimpleNamespace(embedder=SimpleNamespace(embed=lambda *_: None), gallery=SimpleNamespace(search=near)),
        [dict(business_id=cid, plan=plan, kind=kind, upgraded=up) for cid, plan, kind, up, _ in rows])
    assert [matcher.identify(None, None, True, PlanType.LOGIC, column) for column in range(3)] == [
        '872', '99', '103',
    ]
    assert scopes == [{871, 872, 874}, {99, 103}, {103}]
    assert fallback == scopes[:2]


def test_tower_gallery_aliases_match_official_asset_and_grade_without_guessing():
    def gallery(cid, wire, asset, *, upgraded=False):
        return dict(business_id=cid, internal_id=wire, asset_id=asset, upgraded=upgraded, plan='logic', kind='p_idol')

    def official(wire, asset, grade):
        return dict(id=wire, assetId=asset, upgradeCount=grade)

    matcher = FormationThumbnailMatcher(SimpleNamespace(business_ids=(), visual_group_ids=()), None, [
        gallery(1, 'known', 'a'), gallery(2, 'known', 'a', upgraded=True),
        gallery(3, 'wrong-asset', 'b', upgraded=True), gallery(4, 'ambiguous', 'c', upgraded=True),
        gallery(5, 'ambiguous', 'c', upgraded=True), gallery(6, 'wrong-grade', 'd', upgraded=True),
    ])
    master = [official('known', 'a', 0), official('known', 'a', 1), official('wrong-asset', 'c', 1),
              official('ambiguous', 'c', 1), official('wrong-grade', 'd', 0)]
    catalog = dict.fromkeys(('known', 'known+', 'wrong-asset+', 'ambiguous+', 'wrong-grade+'))
    assert matcher.native_card_aliases({'2', '3', '4', '6'}, catalog, PlanType.LOGIC, master) == {
        'known': '1', 'known+': '2',
    }


@pytest.mark.parametrize(('plan_id', 'effect', 'theme', 'plan'), [
    (1, 'ExamLessonBuff', 'lesson_buff', PlanType.SENSE),
    (2, 'ExamCardPlayAggressive', 'aggressive', PlanType.LOGIC),
    (3, 'ExamFullPower', 'full_power', PlanType.ANOMALY),
])
def test_uses_the_selected_idols_plan_and_off_page_basic_cards(plan_id, effect, theme, plan):
    overview = read_formation_overview([text('1/2', 365, 658), text('挑戦する', 357, 1141),
        text('共通曲', 260, 370), text('月村 手毬', 269, 405)])
    char = [{'id': 'ttmr', 'lastName': '月村', 'firstName': '手毬'}]
    card = dict(characterId='ttmr', id='selected', name='共通曲',
                planType=f'ProducePlanType_Plan{plan_id}', examEffectType='ProduceExamEffectType_' + effect)
    wrong = dict(card, characterId='other', id='other')
    deck = [{'id': 'initial_deck-' + theme, 'produceCardIds': ['basic-1', 'basic-2'],
             'produceCardUpgradeCounts': []}]
    identity = resolve_formation_idol(overview, characters=char, idol_cards=[wrong, card], initial_decks=deck)
    assert (identity.idol_card_id, identity.plan) == ('selected', plan)
    assert identity.initial_cards == (('basic-1', 0), ('basic-2', 0))
    assert resolve_formation_idol(overview, characters=char, idol_cards=[wrong], initial_decks=deck) is None
    missing = resolve_formation_idol(overview, characters=char, idol_cards=[card], initial_decks=[])
    assert missing.initial_cards is None


@pytest.mark.parametrize('ellipsis', ('…', '...', '．．．'))
def test_truncated_formation_title_requires_a_unique_prefix_for_the_same_idol(ellipsis):
    prefix = '自己肯定感爆上げ↑↑しゅ'
    overview = read_formation_overview([text('STAGE 21', 96, 61), text('1/2', 360, 673),
        text('挑戦する', 359, 1089), text(prefix + ellipsis, 327, 372), text('藤田 ことね', 279, 404)])
    characters = [{'id': 'fktn', 'lastName': '藤田', 'firstName': 'ことね'}]
    card = dict(characterId='fktn', id='i_card-fktn-3-013', name=prefix + 'きしゅきソング',
                planType='ProducePlanType_Plan1', examEffectType='ProduceExamEffectType_ExamLessonBuff')
    other_idol = dict(card, characterId='other', id='other-idol')
    ambiguous = dict(card, id='other-song', name=prefix + 'が付く別曲')

    def resolve(title, cards):
        return resolve_formation_idol(SimpleNamespace(idol_name=overview.idol_name, card_title=title),
                                     characters=characters, idol_cards=cards, initial_decks=[])

    assert resolve(overview.card_title, [other_idol, card]).idol_card_id == card['id']
    assert resolve(overview.card_title, [other_idol]) is None
    assert resolve(overview.card_title, [card, ambiguous]) is None
    assert resolve(prefix, [card]) is None
    assert resolve(ellipsis, [card]) is None
    assert find_idol_card(overview.idol_name, overview.card_title, characters=characters, idol_cards=[card]) is None


def test_formation_memory_library_adds_only_missing_local_references(monkeypatch, tmp_path):
    size = 8
    num_pixels = size * size * 3
    shared = SimpleNamespace(keys=['shared'], templates=np.zeros((1, num_pixels), dtype=np.float32),
                             masks=np.ones((1, num_pixels), dtype=np.float32), counts=np.array([num_pixels]),
                             names={'shared': 'Shared art'})
    icon_library = Mock(side_effect=lambda **fields: SimpleNamespace(**fields))
    icon_library.load.return_value = shared
    upstream = SimpleNamespace(IconLibrary=icon_library, MIN_SCORE=.5, MIN_MARGIN=.15, _highpass=lambda image: image)
    monkeypatch.setattr(formation, 'hiro_module', lambda _: upstream)
    monkeypatch.setattr(formation, 'hiro_route_file', lambda path: tmp_path / path)
    directory = tmp_path / 'official-art'
    directory.mkdir()
    (directory / 'shared.webp').write_bytes(b'already supplied by Hiro')
    icon = np.zeros((size, size, 4), dtype=np.uint8)
    icon[2:6, 2:6] = (10, 20, 240, 255)
    result, encoded = cv2.imencode('.webp', icon)
    assert result
    (directory / 'local-only.webp').write_bytes(encoded.tobytes())

    library, threshold, margin = formation._formation_memory_icon_library(tmp_path, size)
    assert library.keys == ['shared', 'local-only'] and shared.keys == ['shared']
    np.testing.assert_array_equal(library.templates[:1], shared.templates)
    assert library.counts[1] == 48
    assert np.linalg.norm(library.templates[1]) == pytest.approx(1.)
    assert (threshold, margin) == (.5, .15)
