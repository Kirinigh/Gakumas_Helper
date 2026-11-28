"""Hiro's cultivation scenario fields over Maa's shared live exam executor."""
from __future__ import annotations

import re
import csv
from types import SimpleNamespace
from dataclasses import dataclass

from .upgrade import compact_ocr_text
from .exam_hud import read_lesson_goal, read_exam_ranking, exam_failed_header, lesson_turns_shown
from .item_hud import COMPACT, REGULAR, observe_frame, load_hud_references
from .drink_icons import EVENT_GREEN_JUICE_ID, load_default_drink_icons
from .hiro_runtime import hiro_module
from .idol_road_maa import IdolRoadMaaBattle, _data, read_exam_card_catalog
from .hif_upstream_ui import HIF_SKIP_RECOGNITION
from .idol_road_battle import read_idol_road_live_ranking
from .idol_road_native import NativeTowerRules
from .high_score.kernel import PythonNiaProKernel
from .idol_road_runtime import ATTRIBUTES
from .idol_road_formation import FormationIdol
from .high_score.contracts import PlanType
from .high_score.catalog_v2 import compile_item_catalog, compile_drink_catalog
from .high_score.tower_gnrpa import load_native_tower
from .high_score.gkms_master_forms import PLAN_CODES

_MODES = {'produce-001', 'produce-002', 'produce-003', 'produce-004', 'produce-005'}
_PHASES = {'mid1': 'AUDITION_MID1', 'mid2': 'AUDITION_MID2', 'final': 'AUDITION_FINAL'}
_MASTER_PHASES = {'mid1': 'ProduceStepType_AuditionMid1', 'mid2': 'ProduceStepType_AuditionMid2',
                  'final': 'ProduceStepType_AuditionFinal'}
_ATTRS = {'ボーカル': 'vocal', 'ダンス': 'dance', 'ビジュアル': 'visual'}
_PLANS = {f'ProducePlanType_Plan{code - 1}': PlanType(name) for name, code in PLAN_CODES.items()}


@dataclass
class CultivationEntry:
    produce: str
    idol: str
    max_stamina: int | None
    parameters: tuple[int, int, int] | None = None
    round_key: str | None = None
    audition_number: int | None = None
    lesson_kind: str | None = None


def cultivation_master():
    _, upstream = load_native_tower()
    master = upstream.master()
    return {name: master.rows(name) for name in
            ('Character', 'IdolCard', 'ProduceStepAuditionDifficulty', 'ProduceExamBattleConfig')}


def _text(row):
    return compact_ocr_text(row.text)


def _y(row):
    return row.box[1] + row.box[3] // 2


def read_cultivation_briefing(rows):
    """Hiro's stage and vote tier from the expanded, selected audition."""
    head = ''.join(_text(row) for row in rows if _y(row) < 290)
    if not re.search(r'オーディション|試験', head):
        return None, None
    phase = ('final' if '最終' in head else 'mid2' if re.search(r'[2２]次|中間[2２]', head)
             else 'mid1' if re.search(r'[1１]次|中間', head) else None)
    start = next((row for row in rows if 'オーディション開始' in _text(row)), None)
    if start is None:
        return phase, None
    votes = []
    for row in rows:
        if _y(row) < 700 or not 250 <= row.box[0] + row.box[2] // 2 <= 470:
            continue
        label = _text(row)
        match = re.search(r'([0-9,]+)以上', label)
        if match or '条件なし' in label:
            votes.append((_y(row), int(match[1].replace(',', '')) if match else 0))
    chosen = {n for y, n in votes if 0 < y - _y(start) < 120}
    order = sorted({n for _, n in votes})
    return phase, order.index(chosen.pop()) + 1 if len(chosen) == 1 and (0 in order or len(order) > 1) else None


def _stamina(rows):
    labels = [row for row in rows if '体力' in _text(row)]
    values = set()
    for row in rows:
        if '体力' not in _text(row) and not any(abs(_y(row) - _y(label)) <= 28 for label in labels):
            continue
        match = re.search(r'(?:最大体力|体力)[+：:]?(\d{2,3})$|\d{1,3}[/／](\d{2,3})$', _text(row))
        if match:
            values.add(int(match[1] or match[2]))
    return next(iter(values)) if len(values) == 1 else None


def _parameters(rows):
    values = {}
    for label, attr in _ATTRS.items():
        found = set()
        for row in rows:
            match = re.fullmatch(label + r'[:：]?(\d{1,5})', _text(row))
            if match:
                found.add(int(match[1]))
            elif label == _text(row):
                found.update(int(_text(value)) for value in rows if value.box[0] > row.box[0]
                    and abs(_y(value) - _y(row)) <= 16 and re.fullmatch(r'\d{1,5}', _text(value)))
        if len(found) == 1:
            values[attr] = found.pop()
    return tuple(values[attr] for attr in ATTRIBUTES) if len(values) == 3 else None


def read_cultivation_entry(rows, produce, tables):
    if produce not in _MODES:
        raise ValueError('cultivation_exam_scenario_unknown')
    return CultivationEntry(produce, cultivation_idol(rows, tables), _stamina(rows), _parameters(rows))


def cultivation_idol(rows, tables):
    names = {_text(row).strip('[]【】') for row in rows}
    text = ' '.join(names)
    people = {row['id'] for row in tables['Character']
              if ''.join((row['lastName'] + row['firstName']).split()) in text}
    idols = {row['id'] for row in tables['IdolCard']
             if row['characterId'] in people and any(
                 name == (title := compact_ocr_text(row['name']))
                 or f'({title})' in name or f'「{title}」' in name or f'[{title}]' in name for name in names)}
    if len(idols) != 1:
        raise ValueError('cultivation_exam_launch_idol_unread')
    return idols.pop()


def bind_audition(entry, turns, tables, master, audition_row, score_bonus_permils):
    """Bind the selected idol, mode and exam in Hiro's engine master."""
    idol, = [row for row in tables['IdolCard'] if row['id'] == entry.idol]
    rows = [row for row in tables['ProduceStepAuditionDifficulty']
            if row['id'] == idol['produceStepAuditionDifficultyId'] and row['produceId'] == entry.produce]
    configs = {row['id']: row for row in tables['ProduceExamBattleConfig']}
    phases = (entry.round_key,) if entry.round_key else tuple(_PHASES)
    candidates = []
    for phase in phases:
        for row in rows:
            config = configs.get(row['produceExamBattleConfigId'])
            if (row['stepType'] == _MASTER_PHASES[phase] and config and int(config['turn']) == turns
                    and (entry.audition_number is None or int(row['number']) == entry.audition_number)):
                candidates.append((phase, row))
    if len(candidates) != 1:
        raise ValueError('cultivation_exam_audition_binding_ambiguous')
    phase, local = candidates[0]
    row = audition_row(entry.idol, entry.produce, phase, int(local['number']))
    config = master.battle_config(row['produceExamBattleConfigId'])
    if not config or config['turn'] != turns or entry.parameters is None:
        raise ValueError('cultivation_exam_audition_configuration_unread')
    bonus = score_bonus_permils(row['produceExamBattleConfigId'], entry.parameters)
    if isinstance(bonus, dict):
        bonus = tuple(int(bonus.get(i, bonus.get(str(i)))) for i in (1, 2, 3))
    return dict(step=_PHASES[phase], crit=[int(config[k]) for k in ATTRIBUTES]), tuple(bonus), row


def _lesson_attribute(image):
    attr = hiro_module('vision.hajime').lesson_attr(image)
    return {'Vo': 'vocal', 'Da': 'dance', 'Vi': 'visual'}.get(attr)


def read_lesson_result(rows, attribute):
    """Close a lesson on its actual gained value, including a split score row."""
    titles = [_text(row) for row in rows if _y(row) < 450]
    if not any(re.search(r'レッスン(?:結果|完了)|PERFECT|CLEAR', text) for text in titles):
        return None
    label, = (name for name, value in _ATTRS.items() if value == attribute)
    output = set()
    names = ('獲得スコア', '獲得値', label)
    for row in rows:
        text = _text(row)
        if match := re.fullmatch(rf'(?:獲得スコア[+:：]?|獲得値[+:：]?|{label}\+)([0-9,]+)', text):
            output.add(int(match[1].replace(',', '')))
        if text in names:
            pattern = r'\+[0-9,]+' if text == label else r'\+?[0-9,]+'
            output.update(int(_text(value).lstrip('+').replace(',', ''))
                          for value in rows if value.box[0] > row.box[0] and abs(_y(value) - _y(row)) <= 16
                          and re.fullmatch(pattern, _text(value)))
    return dict(rank=None, score=output.pop()) if len(output) == 1 else None


class CultivationMaaExam(IdolRoadMaaBattle):
    """Reuse the actual card, drink, secondary-choice and receipt loop."""

    hiro_policy_route = True

    def __init__(self, context, verify_input, *, log, entry, algorithm, client_size, score_reader):
        super().__init__(context, verify_input, log=log)
        self.entry, self.algorithm, self.client_size = entry, algorithm, client_size
        self.score_reader = score_reader
        self.lesson = None
        self.lesson_base = None

    def bind_opening(self, image):
        _, upstream = load_native_tower()
        from hiro.search.exam_env import audition_row, lesson_levels_by_clear

        tables = cultivation_master()
        idol, = [row for row in tables['IdolCard'] if row['id'] == self.entry.idol]
        plan = _PLANS[idol['planType']]
        turns = self.remaining_turns(image)
        if turns is None:
            return None
        if self.entry.max_stamina is None:
            self.entry.max_stamina = _stamina(self.ocr(image, roi=(480, 130, 240, 135)))
            if self.entry.max_stamina is None:
                raise ValueError('cultivation_exam_max_stamina_unread')
        self.lesson = self.read_lesson_goal(image)
        gimmick = None
        if self.lesson is not None:
            attr = _lesson_attribute(image)
            if attr is None or self.lesson['goal'] != 'clear' or self.lesson['left'] is None:
                return None
            from .idol_road_timeline import TowerTurnHint

            kinds_to_read = (self.entry.lesson_kind,) if self.entry.lesson_kind else ('normal', 'sp', 'hard')
            levels = [(kind, row) for kind in kinds_to_read
                      for row in lesson_levels_by_clear(self.entry.produce, self.entry.idol, kind, self.lesson['left'], turns)
                      if row['limitTurn'] == turns]
            kinds = {kind for kind, _ in levels}
            if len(kinds) != 1:
                raise ValueError('cultivation_exam_lesson_master_ambiguous')
            kind = kinds.pop()
            step = 1 + ATTRIBUTES.index(attr) * 3 + ('normal', 'sp', 'hard').index(kind)
            exam = dict(step=step, lesson=True, crit=[0, 0, 0])
            bonus = (1000, 1000, 1000)
            self.timeline = {turn: TowerTurnHint(turn, attr) for turn in range(1, turns + 1)}
            self.lesson_base = self.lesson['parameter']
            if self.lesson_base is None:
                return None
        else:
            if self._audition_attribute(image) is None:
                return None
            binding = self.bind_audition_opening(image, turns, tables, upstream, audition_row)
            if binding is None:
                return None
            exam, bonus, row = binding
            gimmick = row.get('produceExamGimmickEffectGroupId') or None
        exam.update(produce=self.entry.produce, idol=self.entry.idol, plan=PLAN_CODES[plan.value])
        self.loadout = SimpleNamespace(idol=FormationIdol(self.entry.idol, plan, None),
            overview=SimpleNamespace(turns=turns, multiplier_permils=bonus, max_stamina=self.entry.max_stamina),
            exam_type=0 if self.lesson is not None else 1)
        self.multipliers = bonus
        drinks_path = _data('p_drinks.csv')
        items_path = _data('p_items.csv')
        cards = read_exam_card_catalog(plan, native=True)
        drinks = compile_drink_catalog(drinks_path, plan_type=plan).drinks
        self.item_catalog = compile_item_catalog(items_path, plan_type=plan)
        self.kernel = PythonNiaProKernel(cards, drinks, self.item_catalog.items)
        observed_items = self._items(image, {row['id']: row['name'] for row in self._csv(items_path)
                                            if row['mode'] == 'stage' and row['id'] in self.kernel.items})
        self.native_rules = NativeTowerRules(self.kernel, SimpleNamespace(identity=gimmick) if gimmick else None,
                                              items=observed_items)
        self.art = self._art()
        self.drink_reader = load_default_drink_icons(_data('hif_drink_icons'),
            (*tuple(row['id'] for row in self._csv(drinks_path)), EVENT_GREEN_JUICE_ID))
        self.deck_ids = ()
        self.unread_formation_cards = None
        return exam

    def read_lesson_goal(self, image):
        return read_lesson_goal(image, self.ocr)

    def bind_audition_opening(self, image, turns, tables, upstream, audition_row):
        return bind_audition(self.entry, turns, tables, upstream.master(), audition_row,
                             upstream.score_bonus_permils)

    @staticmethod
    def _csv(path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return tuple(csv.DictReader(stream))

    def _items(self, image, metadata):
        refs, crosswalk = load_hud_references(_data('hif_item_icons'), frozenset())
        layouts = [observe_frame(self.context, image, layout, refs, crosswalk) for layout in (REGULAR, COMPACT)]
        best = max(layouts, key=lambda rows: sum(bool(row['accepted_asset']) for row in rows))
        for row in best:
            found = set(row['candidates']) & metadata.keys()
            if row['accepted_asset'] and len(found) == 1:
                cid, = found
                self.item_uses[cid] = row['remaining_uses']
        return {cid: metadata[cid] for cid in self.item_uses}

    def _audition_attribute(self, image):
        return self.read_audition_attribute(image, (95, 38, 172, 62))

    def read_exam_position(self, image):
        if self.lesson is not None:
            lesson = read_lesson_goal(image, self.ocr)
            if lesson is None or lesson['parameter'] is None:
                return None, None
            return self.timeline[1].attribute, lesson['parameter'] - self.lesson_base
        rank = read_idol_road_live_ranking(image, self.ranking_number)
        score = rank['score'] if rank and rank['score'] is not None else self.score_reader(self.context, image)
        return self._audition_attribute(image), score

    def result_page_ready(self, image):
        return not (self.recognition('ProduceRecognitionHealthFlag', image)
                    or (self.lesson is not None and lesson_turns_shown(image, self.remaining_turns)))

    def read_result_page(self, image):
        if self.lesson is None:
            return read_exam_ranking(image, self.ocr) or self.read_unscored_failure(image)
        return read_lesson_result(self.ocr(image), self.timeline[1].attribute)

    def read_unscored_failure(self, image):
        if (self.lesson is None and self.entry.produce in _MODES
                and exam_failed_header(image, self.ocr)):
            return dict(rank=None, score=None, failed=True)
        return None

    def run_cultivation(self, *, hiro_session=None):
        def can_open(image):
            if not self.target('ProduceRecognitionSkipRound', image, HIF_SKIP_RECOGNITION):
                return None
            return self.bind_opening(image)

        exam = self.poll(can_open, 'cultivation-opening-master')
        self.live_screen = self.poll(self.observe_live, 'cultivation-opening-hud')
        return self.run_live_exam(exam=exam, hiro_label=self.exam_label(exam), hiro_session=hiro_session)

    def exam_label(self, exam):
        if exam.get('lesson'):
            attr, kind = divmod(exam['step'] - 1, 3)
            code = ('Vo', 'Da', 'Vi')[attr]
            options = [{'oikomi': True}] if kind == 2 else []
            return hiro_module('runtime.hajime').lesson_label(code, SimpleNamespace(sp=kind == 1), options)
        phase = next(phase for phase, key in _PHASES.items() if key == exam['step'])
        if self.entry.produce in hiro_module('runtime.nia_sample').MODES.values():
            suffix = f'#audition-{self.entry.audition_number}' if self.entry.audition_number is not None else ''
            return f'nia-{phase}{suffix}'
        return f'hajime-{phase}'
