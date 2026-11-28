"""The visible Idol Road turn table and its stage effects."""

import re
import unicodedata
from dataclasses import dataclass

import numpy as np

if __package__.startswith('agent.'):
    from agent.utils.effect_text import normalize_score_effect_text
else:
    from utils.effect_text import normalize_score_effect_text

from .turn_panel import ATTRIBUTES


@dataclass(frozen=True)
class TowerTurnHint:
    turn: int
    attribute: str
    description: str = ''


@dataclass(frozen=True)
class TowerTimelinePage:
    current_turn: int
    rows: tuple[TowerTurnHint, ...]
    thumb: tuple[int, int] | None

    @property
    def at_top(self):
        return self.thumb is None or self.thumb[0] <= 307

    @property
    def at_bottom(self):
        return self.thumb is None or self.thumb[1] >= 1140


@dataclass(frozen=True)
class TowerFormationEffectsPage:
    layer: int
    rows: tuple[TowerTurnHint, ...]
    at_bottom: bool
    close: tuple[int, int]


def _spans(pixels, first):
    edges = np.diff(np.r_[False, pixels, False].astype(np.int8))
    return tuple((int(a + first), int(b + first)) for a, b in
                 zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _text_in_reading_order(rows):
    lines = []
    for text, x, y in sorted(rows, key=lambda row: row[2]):
        if not lines or y - lines[-1][0] > 13:
            lines.append((y, []))
        lines[-1][1].append((x, text))
    return ''.join(text for _, line in lines for _, text in sorted(line))


def read_formation_effects_page(image, ocr_rows):
    """Read full effect rows from the pre-challenge Support/Trouble drawer."""
    if image.shape != (1280, 720, 3):
        return None
    tokens = [(unicodedata.normalize('NFKC', row.text).replace(' ', ''), x, y + h / 2,
               (x + w // 2, y + h // 2)) for row in ocr_rows if row.score >= .90
              for x, y, w, h in (row.box,)]
    heading_rows = [(text, y) for text, _, y, _ in sorted(tokens, key=lambda row: row[1]) if 80 <= y <= 170]
    heading = ''.join(text for text, _ in heading_rows)
    if not re.search(r'応援[/ノ・]?トラブル', heading):
        return None
    # As with Hiro's modal readers, scope the body from the displayed title.
    # The stage line moves down when the effect drawer uses its taller header.
    title_y = max(y for _, y in heading_rows)
    levels = {int(match[1]) for text, x, y, _ in tokens if x < 255 and title_y + 25 <= y <= title_y + 130
              and (match := re.fullmatch(r'ステージ(\d+)', text))}
    closes = {point for text, _, y, point in tokens if text == '閉じる' and 1090 <= y <= 1220}
    if len(levels) != 1 or len(closes) != 1:
        return None
    layer, = levels
    close, = closes
    first, last = 234, 1066
    body = image[first:last, 61:658].astype(np.int16)
    gray = (body.min(axis=2) >= 180) & (body.max(axis=2) < 235) & (np.ptp(body, axis=2) < 8)
    borders = _spans(gray.mean(axis=1) >= .87, first)
    rows = []
    for text, x, y, _ in tokens:
        if x > 90 or not first <= y < last or not (label := re.fullmatch(r'([1-9]\d?)ターン目', text)):
            continue
        following = [top for top, _ in borders if y < top]
        if len(following) < 2 or following[0] - y > 45:
            continue
        divider, bottom = following[:2]
        description = _text_in_reading_order([(word, px, cy) for word, px, cy, _ in tokens
                                               if px >= 150 and divider < cy < bottom])
        if description:
            rows.append(TowerTurnHint(int(label[1]), '', description))
    strip = image[first:last, 679].astype(np.int16)
    thumb = np.flatnonzero((strip.min(axis=1) < 180) & (np.ptp(strip, axis=1) < 26))
    at_bottom = not len(thumb) or int(thumb[-1]) >= last - first - 8
    return TowerFormationEffectsPage(layer, tuple(sorted(rows, key=lambda row: row.turn)), at_bottom, close)


def read_tower_timeline_page(image, ocr_rows):
    """Bind the effect text to complete row borders, including tall effect rows."""
    if image.shape != (1280, 720, 3):
        return None
    tokens = [(unicodedata.normalize('NFKC', row.text).replace(' ', ''),
               row.box[0], row.box[1] + row.box[3] / 2) for row in ocr_rows if row.score >= .90]
    turns = [int(match[1]) for text, x, y in tokens if x < 250 and 126 <= y < 172
             and (match := re.fullmatch(r'現在(\d+)ターン目', text))]
    if len(turns) != 1 or not any(re.fullmatch(r'\d+位以上で合格', text) and 166 < y < 207
                                  for text, _, y in tokens):
        return None
    current, = turns
    first, last = 303, 1147
    strip = image[first:last, 689:692].astype(np.int16)
    solid = ((strip.min(axis=(1, 2)) >= 90) & (strip.max(axis=(1, 2)) <= 180)
             & (np.ptp(strip, axis=(1, 2)) <= 32) & (np.ptp(strip, axis=1).max(axis=1) <= 2))
    thumbs = [span for span in _spans(solid, first) if span[1] - span[0] >= 30]
    if len(thumbs) > 1:
        return None
    body = image[first:last, 60:662].astype(np.int16)
    gray = (body.min(axis=2) >= 190) & (body.max(axis=2) < 245) & (np.ptp(body, axis=2) < 8)
    borders = [span for span in _spans((gray[:, :250].mean(axis=1) >= .97)
                                      & (gray[:, -250:].mean(axis=1) >= .97), first) if span[1] - span[0] <= 5]
    rows = []
    for top, bottom in zip(borders, borders[1:]):
        if bottom[1] - top[0] < 72:
            continue
        inside = [(text, x, y) for text, x, y in tokens if top[1] < y < bottom[0]]
        labels = [text for text, x, y in inside if 35 <= x < 180 and y <= top[0] + 66
                  and (text == '現在のターン' or re.fullmatch(r'[1-9]\d?ターン[前後]', text))]
        attrs = [(text, y) for text, x, y in inside if 235 <= x <= 300 and text in ATTRIBUTES
                 and y <= top[0] + 65]
        if len(labels) != 1 or len(attrs) != 1:
            continue
        label, = labels
        (attribute, y), = attrs
        offset = re.fullmatch(r'(\d+)ターン([前後])', label)
        turn = current + (int(offset[1]) * (1 if offset[2] == '後' else -1) if offset else 0)
        description = _text_in_reading_order([(text, x, cy) for text, x, cy in inside if x >= 150 and cy > y + 27])
        rows.append(TowerTurnHint(turn, ATTRIBUTES[attribute], description))
    return TowerTimelinePage(current, tuple(rows), thumbs[0] if thumbs else None)


_EFFECT_ICON_READS = re.compile(
    r'(?<![A-Za-z0-9])(?:'
    r'\(?\s*A\s*\)?\s*(?=アクティブスキルカード)'
    r'|\(?\s*M\s*\)?\s*(?=集中|メンタルスキルカード)'
    r'|\(?\s*R\s*\)?\s*(?=スコア上昇)'
    r')'
    r'|(?<=[、,])回(?=スキルカード使用数追加\+\d+$)'
)

_EFFECT_FACTS = re.compile(
    r'\d+|[+%−-]|以上|以下|未満|がない|がある|'
    r'好印象|やる気|集中|好調|絶好調|全力|元気|体力|スコア|'
    r'アクティブ|メンタル|SSR|SR|ターン開始|ターン終了|'
    r'使用不可|使用可能|増加|減少|消費|回復'
)


def _normalize_effect(text, *, observed):
    text = unicodedata.normalize('NFKC', text).translate(str.maketrans('增减态', '増減態'))
    text = normalize_score_effect_text(text)
    if not observed:
        return text
    text = text.replace('ァクティブスキルカード', 'アクティブスキルカード')
    # These glyphs are OCR readings of icons next to the named effect. Keep
    # the text of the effect itself and meaningful Latin labels such as SSR.
    return _EFFECT_ICON_READS.sub('', text)


def tower_field_rule(text):
    """Return only rules whose condition and amount both appear in the UI."""
    text = _normalize_effect(text, observed=True)
    if re.fullmatch(r'(?:直前に)?アクティブスキルカードを使用していた場合[、,]メンタルスキルカード使用不可\(1ターン\)', text):
        return 'block_mental', 1, 'active'
    if re.fullmatch(r'(?:直前に)?メンタルスキルカードを使用していた場合[、,]アクティブスキルカード使用不可\(1ターン\)', text):
        return 'block_active', 1, 'mental'
    if match := re.fullmatch(r'やる気が(\d+)以上の場合[、,]以降[、,]ターン終了時[、,]好印象\+(\d+)', text):
        return 'review_at_end', int(match[2]), int(match[1])
    if match := re.fullmatch(r'好印象が(\d+)以上の場合[、,]スキルカード使用数追加\+(\d+)', text):
        return 'extra_actions', int(match[2]), int(match[1])
    if match := re.fullmatch(r'好印象が(\d+)以上の場合[、,]ターン追加\+(\d+)', text):
        return 'extra_turns', int(match[2]), int(match[1])
    if match := re.fullmatch(r'好印象が(\d+)以上の場合[、,]好印象(\d+)倍', text):
        return 'review_multiple', int(match[2]), int(match[1])
    if match := re.fullmatch(r'(?:好印象が(\d+)以上の場合[、,]?)?好印象\+(\d+)', text):
        return 'review', int(match[2]), int(match[1]) if match[1] else None
    if match := re.fullmatch(r'スコアが([\d,]+)未満の場合[、,]?(?:気まぐれ|スランプ)(\d+)ターン', text):
        return 'slump', int(match[2]), int(match[1].replace(',', ''))
    if match := re.fullmatch(r'好印象が(\d+)以下の場合[、,](?:気まぐれ|スランプ)(\d+)ターン', text):
        return 'review_slump', int(match[2]), int(match[1])
    if match := re.fullmatch(r'スコアが([\d,]+)以上の場合[、,]スコア上昇量増加(\d+)%\((\d+)ターン\)', text):
        return 'score_buff', int(match[2]), (int(match[1].replace(',', '')), int(match[3]))
    if match := re.fullmatch(r'(?:好印象が(\d+)以上の場合[、,]?)?好印象の(\d+)%分スコア上昇', text):
        return 'review_score', int(match[2]), int(match[1]) if match[1] else None
    return None


def match_formation_effects(rows, variants):
    """Bind the visible effects to one specific schedule from the master."""
    def normalize(text, *, observed):
        text = _normalize_effect(text, observed=observed).replace('気まぐれ', 'スランプ')
        # The master labels ExamStaminaReduceFix as 体力消費; the game adds 固定.
        text = text.replace('固定体力消費', '体力消費')
        text = re.sub(r'タ\s*[-‐‑‒–—―−一ー]\s*ン', 'ターン', text)
        text = re.sub(r'^\d(?=好印象\d+倍$)', '', text)
        # OCR can omit words. Keep numbers intact so 18 never matches 85 or 118.
        return re.findall(r'\d+|[^\d\s、,。]', text)

    def is_subsequence(visible, complete):
        remaining = iter(complete)
        return all(char in remaining for char in visible)

    visible = {turn: tokens for turn, row in rows.items() if (tokens := normalize(row.description, observed=True))}
    if not visible:
        return None
    eligible = {key: {turn: normalize(variant[turn], observed=False) for turn in visible}
                for key, variant in variants.items() if visible.keys() <= variant.keys()}

    def matching(fold):
        return [key for key, variant in eligible.items()
                if all(is_subsequence(map(fold, tokens), map(fold, variant[turn]))
                       for turn, tokens in visible.items())]

    matches = matching(lambda token: token)
    if matches:
        return matches[0] if len(matches) == 1 else None

    def fold(token):
        # Like Hiro's OCR comparison, fold visually confusable kana at match time.
        return unicodedata.normalize('NFD', token).replace('\u309a', '\u3099')

    matches = matching(fold)
    if matches:
        return matches[0] if len(matches) == 1 else None

    supporting, unread, matched_text_rows = [], {}, 0
    for turn, tokens in visible.items():
        candidates = {key for key, variant in eligible.items()
                      if is_subsequence(map(fold, tokens), map(fold, variant[turn]))}
        if candidates:
            supporting.append(candidates)
            matched_text_rows += any(token.isalpha() for token in tokens)
        else:
            unread[turn] = tokens
    if matched_text_rows < 2:
        return None
    common = set.intersection(*supporting)
    if len(common) != 1:
        return None
    key, = common
    for turn, tokens in unread.items():
        # A broken OCR row can still contain a clearly read condition or amount.
        observed_facts = _EFFECT_FACTS.findall(''.join(tokens))
        master_facts = _EFFECT_FACTS.findall(''.join(eligible[key][turn]))
        if not is_subsequence(observed_facts, master_facts):
            return None
    return key


def complete_formation_effects(rows, variants):
    """Fill the schedule when the visible rows identify one master variant."""
    variants = dict(enumerate(variants))
    key = match_formation_effects(rows, variants)
    if key is None:
        return rows
    return {turn: TowerTurnHint(turn, rows[turn].attribute if turn in rows else '', description)
            for turn, description in variants[key].items()}
