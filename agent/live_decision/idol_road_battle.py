"""Idol Road opening and results for the Maa battle decision interface."""

import re
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from .exam_hud import read_exam_rank_label
from .idol_road_formation import (
    FormationCard,
    FormationIdol,
    FormationOverview,
    read_formation_deck,
    formation_deck_at_edge,
    resolve_formation_idol,
    read_formation_overview,
    has_fourth_formation_memory,
)
from .high_score.decision_box import open_decision_box
from .high_score.gkms_master_forms import PLAN_CODES, card_form
from .high_score.gkms_shadow_adapter import ExamContext, TurnPlayHistory
from .high_score.gkms_shadow_mappings import CARD_FORMS


def _ranking_columns(player_rank):
    right = 710
    columns = []
    for rank in range(6, 0, -1):
        width = 127 if rank == player_rank else 69
        columns.append((right - width, right))
        right -= width + 2 + (13 if rank == 4 else 0)
    return columns[::-1]


def read_idol_road_live_ranking(image, read_number):
    """Read the player's live score using Hiro's six-column Tower geometry."""
    if image.shape != (1280, 720, 3):
        return None
    layouts = [_ranking_columns(rank) for rank in range(1, 7)]
    # DMM outlines the player's wide column even while its fill changes.
    # A cyan animation can briefly fill two adjacent rival columns.
    white = (image[50:120].min(axis=2) >= 230).mean(axis=0)
    framed = []
    for index, columns in enumerate(layouts):
        left, right = columns[index]
        if white[left - 3:left + 11].max() >= .9 and white[right - 4:right + 5].max() >= .9:
            framed.append(index)
    if len(framed) == 1:
        player, = framed
    else:
        lane = cv2.cvtColor(image[50:120], cv2.COLOR_BGR2HSV)
        hue, saturation, value = (lane[..., channel] for channel in range(3))
        cyan = (85 <= hue) & (hue <= 105) & (saturation >= 120) & (value >= 150)
        ratios = [np.mean(cyan[:, columns[i][0] + 4:columns[i][1] - 40])
                  for i, columns in enumerate(layouts)]
        player = int(np.argmax(ratios))
        if ratios[player] < .15:
            return None
    scores = []
    for index, (left, right) in enumerate(layouts[player]):
        start, end = (left + 2, right + 8) if index == player else (left + 6, left + 80)
        scores.append(read_number(image, (start, 126, min(720, end) - start, 42)))
    if scores[player] is None and not any(scores):
        left = layouts[player][player][0]
        if read_number(image, (left + 7, 128, 41, 35)) == 0:
            scores[player] = 0
    for index in range(len(scores) - 1):
        score, following = scores[index:index + 2]
        if score is not None and following is not None and score < following:
            scores[index] = None
    return {'rank': player + 1, 'score': scores[player]}


def read_idol_road_result(rows, *, idol_name=None):
    """Read the displayed player's rank and points on Hiro's ranking layout."""
    tokens = [(re.sub(r'\s+', '', row.text), row.box[0] + row.box[2] / 2, row.box[1] + row.box[3] / 2)
              for row in rows]
    badges = {int(match[1]) for text, x, y in tokens if x < 200 and 130 <= y <= 260
              and (match := re.fullmatch(r'([1-6])位?', text))}
    scores = sorted((y, int(match[1].replace(',', ''))) for text, _, y in tokens if 260 < y < 1080
                    and (match := re.fullmatch(r'(\d+|\d{1,3}(?:,\d{3})+)[Pp][tTlI1]?', text)))
    if not scores:
        return None
    labels = [(y, rank) for text, x, y in tokens if x < 150 and 260 < y < 1080
              and (rank := read_exam_rank_label(text)) is not None]
    if not badges and idol_name and any('不合格' in text and x < 230 and 130 <= y <= 260
                                        for text, x, y in tokens):
        # Hiro reads the idol's own row on failure, where the top-left rank
        # badge is replaced by 不合格 and the last rival may be off-screen.
        name = re.sub(r'\s+', '', idol_name)
        own_rows = [y for text, x, y in tokens if text == name and 180 < x < 500 and 300 < y < 1080]
        if len(own_rows) == 1:
            own, = own_rows
            ranks = [rank for y, rank in labels if 4 <= rank <= 6 and abs(y - own) < 52]
            points = [score for y, score in scores if 0 < y - own < 70]
            if len(ranks) == len(points) == 1:
                return {'rank': ranks[0], 'score': points[0]}
        return None
    if len(badges) != 1:
        return None
    rank, = badges
    rank_rows = [y for y, found in labels if found == rank]
    bound = [(y, score) for y, score in scores if any(abs(y - ry) < 55 for ry in rank_rows)]
    if len(bound) == 1:
        return {'rank': rank, 'score': bound[0][1]}
    if not rank_rows and len(scores) == 6:
        # With all six entries present, the game lists them in rank order.
        return {'rank': rank, 'score': scores[rank - 1][1]}
    return None


def read_idol_road_result_frame(ocr, image, *, idol_name=None, hiro=False):
    """Use the same player score columns in every Tower strategy."""
    labels = tuple(ocr(image, roi=(0, 130, 200, 950)))
    scores = tuple(ocr(image, roi=(206, 304, 153, 781)))
    if hiro:
        from .hiro_runtime import hiro_module

        buttons = tuple(ocr(image, roi=(0, 1080, 720, 168)))
        tokens = [(''.join(row.text.split()), (x, y, x + w, y + h))
                  for row in labels + scores + buttons for x, y, w, h in (row.box,)]
        page = hiro_module('vision.tower').tower_page(image, tokens)
        if page and page[0] == 'TOWER_RANKING':
            fields = page[1]
            if fields.get('failed'):
                own = [row for row in fields['rows'] if row['name'] == re.sub(r'\s+', '', idol_name or '')
                       and 4 <= row['rank'] <= 6]
            else:
                own = [row for row in fields['rows'] if row['rank'] == fields['my_rank']]
            if len(own) == 1:
                return dict(rank=own[0]['rank'], score=own[0]['score'])
    if (result := read_idol_road_result(labels + scores, idol_name=idol_name)) is not None:
        return result
    rows = tuple(ocr(image))
    return (read_idol_road_result(rows, idol_name=idol_name)
            or read_idol_road_result(tuple(row for row in rows if row.box[0] < 150) + scores, idol_name=idol_name))


@dataclass(frozen=True)
class IdolRoadLoadout:
    overview: FormationOverview
    idol: FormationIdol
    cards: tuple[FormationCard, ...]


def read_idol_road_loadout(image, *, ocr, show_page, scroll_deck, identify, characters, idol_cards, initial_decks,
                          on_overview=None):
    """Follow Hiro's two-page formation read inside the existing Maa task.

    ``show_page(point, number)`` uses the task's normal input and waits for
    that page, returning its current image. The final page is challenge-ready.
    With a fourth memory, ``scroll_deck(toward_bottom)`` reveals the other rows.
    Card art is identified in the actual plan, independently of the policy.
    """
    overview = read_formation_overview(ocr(image))
    if overview is None:
        raise ValueError('idol_road_formation_not_visible')
    if overview.page == 2:
        image = show_page(overview.previous_page, 1)
        overview = read_formation_overview(ocr(image))
    if overview is None or overview.page != 1:
        raise ValueError('idol_road_overview_not_visible')
    idol = resolve_formation_idol(overview, characters=characters, idol_cards=idol_cards,
                                  initial_decks=initial_decks)
    if idol is None:
        raise ValueError('idol_road_formation_idol_unresolved')
    if on_overview is not None:
        on_overview(image, overview, idol)
    image = show_page(overview.next_page, 2)
    deck_page = read_formation_overview(ocr(image))
    if (deck_page is None or deck_page.page != 2
            or (overview.layer is not None and deck_page.layer is not None and overview.layer != deck_page.layer)):
        raise ValueError('idol_road_formation_deck_not_visible')
    plan = deck_page.recommended_plan or overview.recommended_plan or idol.plan

    def identify_card(frame, box, upgraded, column):
        return identify(frame, box, upgraded, plan, column)

    if has_fourth_formation_memory(image):
        if formation_deck_at_edge(image, True):
            top, bottom = scroll_deck(False), image
        else:
            top = image if formation_deck_at_edge(image, False) else scroll_deck(False)
            bottom = scroll_deck(True)
        cards = (read_formation_deck(top, identify_card, with_fourth=True)
                 + read_formation_deck(bottom, identify_card, after_scroll=True))
    else:
        cards = read_formation_deck(image, identify_card)
    return IdolRoadLoadout(overview, idol, cards)


def _wire_card(kernel, card_id, upgraded, *, native=False):
    form = CARD_FORMS.get(card_id) or card_form(kernel.cards.get(card_id), upgraded, native=native)
    if form is None or form[1] != int(upgraded):
        raise ValueError(f'idol_road_card_form_unresolved:{card_id}')
    return form[:2]


class TowerExamContext(ExamContext):
    """UINet accepts an observed prefix; the planner can use a forecast."""

    use_host_legality = True

    def __init__(self, *args, observed_types, formation, rust_stage=None, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'observed_types', observed_types)
        object.__setattr__(self, 'formation', formation)
        object.__setattr__(self, 'rust_stage', rust_stage)

    def observation(self, state, plan_type):
        from dataclasses import replace

        prefix = tuple(self.observed_types)
        if len(prefix) != state.turn:
            raise ValueError('idol_road_missing_observed_turn_attribute')
        attrs = {'vocal': 1, 'dance': 2, 'visual': 3}
        forecast = tuple(attrs[value] for value in state.turn_attributes)
        round_types = prefix[:-1] + forecast
        result = ExamContext.observation(replace(self, turn_types=round_types, observed_types=self.observed_types,
            formation=self.formation, rust_stage=self.rust_stage), state, plan_type)
        if prefix[-1] != forecast[0]:
            raise ValueError('idol_road_attribute_changed_before_decision')
        result['types'] = list(prefix)
        return result


def open_idol_road_battle(algorithm, *, loadout, kernel, opening, planning_seed, deadline, model=None,
                         observed_types=None, rust_stage=None):
    """Bind that formation to the live first HUD and open the chosen algorithm."""
    state = opening.state
    info = loadout.overview
    if (opening.plan_type is not loadout.idol.plan or state.turn != 1
            or state.turns_remaining != info.turns or state.max_stamina != info.max_stamina):
        raise ValueError('idol_road_opening_differs_from_formation')
    if (info.multiplier_permils is None or len(state.turn_attributes) != info.turns
            or loadout.idol.initial_cards is None
            or any(card.card_id is None for card in loadout.cards)):
        raise ValueError('idol_road_opening_data_incomplete')
    # The formation cards and the two off-page basic cards are separate
    # physical instances. The first hand removes instances from this inventory.
    expected = Counter(_wire_card(kernel, card.card_id, card.upgraded) for card in loadout.cards)
    expected.update(loadout.idol.initial_cards)
    observed = Counter(_wire_card(kernel, card.card_id, card.upgraded) for card in state.deck.all_cards)
    if observed != expected:
        raise ValueError('idol_road_opening_deck_differs_from_formation')
    attrs = {'vocal': 1, 'dance': 2, 'visual': 3}
    context_type = ExamContext if observed_types is None else TowerExamContext
    extra = {} if observed_types is None else dict(observed_types=observed_types, formation=loadout, rust_stage=rust_stage)
    context = context_type('tower', None, PLAN_CODES[opening.plan_type.value],
        tuple(attrs[value] for value in state.turn_attributes), info.multiplier_permils, (0, 0, 0),
        loadout.idol.idol_card_id, f'Idol Road formation and opening {opening.snapshot_id}', **extra)
    context.observation(state, opening.plan_type)
    upgrades = {card_id: card.upgrade_card_id for card_id, card in kernel.cards.items()
                if card.upgrade_card_id in kernel.cards}
    return open_decision_box(algorithm, kernel=kernel, opening=opening, context=context,
        history=TurnPlayHistory(state, (), context.source, upgrade_ids=upgrades), planning_seed=planning_seed, deadline=deadline,
        model=model)
