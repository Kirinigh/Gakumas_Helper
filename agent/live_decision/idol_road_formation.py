"""Read the Idol Road formation that the game shows before a challenge.

Screen layout reference: hiro-gakumas-hero 8380622a, vision/tower.py.
The caller supplies Maa OCR and the existing card-art recognizer.
"""

import re
import csv
import json
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from agent.utils.runtime_paths import resolve_runtime_path
from agent.card_selection.embedding import EmbeddingCardRecognizer
from agent.arena_winrate.badge_reference import BadgeReferenceGallery

from .item_hud import rank_icons, load_hud_references
from .hiro_runtime import hiro_module, hiro_route_file
from .high_score.contracts import PlanType

_PLANS = {
    'ProducePlanType_Plan1': PlanType.SENSE,
    'ProducePlanType_Plan2': PlanType.LOGIC,
    'ProducePlanType_Plan3': PlanType.ANOMALY,
}
_RECOMMENDED_PLANS = {
    '集中': PlanType.SENSE, '好調': PlanType.SENSE,
    '好印象': PlanType.LOGIC, 'やる気': PlanType.LOGIC,
    '全力': PlanType.ANOMALY, '強気': PlanType.ANOMALY, '温存': PlanType.ANOMALY,
}
_SOURCE_KIND = {'pIdol': 'p_idol', 'support': 'support', 'produce': 'other', 'default': 'basic'}
_SLOT_KINDS = (frozenset({'p_idol'}), frozenset({'other', 'support'}),
               *(frozenset({'other'}) for _ in range(4)))
_DECK_FOR_EXAM_EFFECT = {
    'ExamLessonBuff': 'lesson_buff',
    'ExamParameterBuff': 'parameter_buff',
    'ExamReview': 'review',
    'ExamCardPlayAggressive': 'aggressive',
    'ExamConcentration': 'concentration',
    'ExamFullPower': 'full_power',
}


@dataclass(frozen=True)
class FormationOverview:
    layer: int | None
    idol_name: str | None
    card_title: str | None
    turns: int | None
    multiplier_permils: tuple[int, ...] | None
    max_stamina: int | None
    page: int
    next_page: tuple[int, int] | None
    previous_page: tuple[int, int] | None
    recommended_plan: PlanType | None = None


@dataclass(frozen=True)
class FormationCard:
    row: int
    column: int
    card_id: str | None
    upgraded: bool


@dataclass(frozen=True)
class FormationIdol:
    idol_card_id: str
    plan: PlanType
    initial_cards: tuple[tuple[str, int], ...] | None


def _only(values):
    unique = set(values)
    return next(iter(unique)) if len(unique) == 1 else None


def read_formation_overview(rows):
    """Use one canonical 720×1280 OCR result from the active formation page."""
    boxes = tuple((row['text'].strip(), x + w // 2, y + h // 2)
                  for row in rows for x, y, w, h in (row['box'],))
    page = _only((int(text[0]), x, y) for text, x, y in boxes
                 if re.fullmatch(r'[12]\s*[/／]\s*2', text) and 280 <= x <= 440 and 600 <= y <= 720)
    labels = {text for text, _, y in boxes if y <= 350}
    # As in Hiro, the formation headers identify the page even if the button text is obscured.
    support_label = any(re.fullmatch(r'応援\s*[/／]\s*トラ[ブプ]ル', text) and 0 <= y <= 60
                        for text, _, y in boxes)
    formation_header = support_label and 'ターン' in labels and bool({'審査基準', '審查基準'} & labels)
    if page is None or not (formation_header or any(text == '挑戦する' and y >= 1000 for text, _, y in boxes)):
        return None
    number, cx, cy = page
    layer = _only(int(match[1]) for text, _, _ in boxes if (match := re.fullmatch(r'STAGE\s*(\d+)', text)))
    heading = _only(y for text, x, y in boxes if text == 'おすすめ効果' and 35 <= x <= 215 and 935 <= y <= 980)
    recommended_plan = _only(_RECOMMENDED_PLANS[text] for text, x, y in boxes
                             if heading is not None and 55 <= x <= 215 and 15 <= y - heading <= 62
                             and text in _RECOMMENDED_PLANS)
    if number == 2:
        return FormationOverview(layer, None, None, None, None, None, number, None, (cx - 97, cy), recommended_plan)

    def number_beside(label, label_y, number_y):
        return _only(int(value) for text, x, y in boxes if text == label and label_y[0] <= y <= label_y[1]
                     for value, nx, ny in boxes if value.isdecimal() and abs(nx - x) < 30 and number_y(y, ny))

    turns = number_beside('ターン', (60, 130), lambda _, y: 30 <= y <= 80)
    stamina = number_beside('体力', (500, 600), lambda y, ny: 15 <= ny - y <= 50)
    rows_by_height = sorted((y, int(match[1]) * 10) for text, x, y in boxes if x > 560
                            and (match := re.fullmatch(r'\D?(\d{2,5})[%％]', text)))
    multipliers = tuple(value for _, value in rows_by_height) if len(rows_by_height) == 3 else None
    idol = _only(text for text, x, y in boxes if 180 <= x <= 400 and 390 <= y <= 425)
    title = _only(text for text, x, y in boxes if 180 <= x <= 330 and 360 <= y < 390)
    return FormationOverview(layer, idol, title, turns, multipliers, stamina, number, (cx + 97, cy), None, recommended_plan)


def _card_row_top(image, expected, first_x=154, *, through=10):
    """Align the six thumbnails by their shared upper border."""
    start, end = expected - 23, expected + through
    gray = cv2.cvtColor(image[start:end], cv2.COLOR_BGR2GRAY)
    edge = np.stack([(gray[:, x + 20:x + 62] < 155).mean(axis=1) for x in range(first_x, first_x + 546, 91)])
    edge = (edge > .7).sum(axis=0) >= 3
    endings = np.flatnonzero(edge[1:-1] & edge[:-2] & ~edge[2:]) + start + 1
    return int(min(endings, key=lambda y: abs(y - expected))) if len(endings) else expected


def _deck_row_tops(image):
    shift = _card_row_top(image, 471) - 471
    return (375 + shift, 471 + shift, 567 + shift)


def has_fourth_formation_memory(image):
    """Detect the extra selected Sub memory by the attribute bar below its portrait."""
    hue, saturation, value = cv2.cvtColor(image[960:1015, 260:457], cv2.COLOR_BGR2HSV).transpose(2, 0, 1)
    bright = (saturation >= 150) & (value >= 175)
    blue = (bright & (hue >= 95) & (hue <= 125)).sum(axis=1)
    amber = (bright & (hue >= 10) & (hue <= 34)).sum(axis=1)
    return bool(((blue >= 12) & (amber >= 16)).any())


def formation_deck_at_edge(image, toward_bottom):
    """Locate the deck scrollbar thumb at the requested end of the track."""
    edge = 622 if toward_bottom else 384
    rail = image[edge:edge + 16, 705:709]
    return bool(((rail.max(axis=2) < 182) & (rail.min(axis=2) > 98)
                 & (np.ptp(rail, axis=2) <= 18)).mean() >= .65)


def _active_formation_cell(cell):
    if cell[..., 2].mean() < 160:
        return False  # Hiro's brightness test for dimmed duplicates and restrictions.
    # Empty placeholders are light gray. Inspect the interior so that the
    # rounded border and nearby cards do not count as card artwork.
    interior = cell[9:-9, 9:-9]
    low, high = np.percentile(interior[..., 2], (10, 90))
    return high - low > 8 or (interior[..., 1] > 12).mean() >= .02


def read_formation_deck(image, identify, *, with_fourth=False, after_scroll=False):
    """Read the active card cells on formation page 2 in formation order.

    ``identify(image, box, upgraded, column)`` adapts the arena matcher to
    the small card-art crop. It returns a local exact card ID, or None when
    unread. Duplicate instances and unread bright cells retain their slots.
    """
    if image.shape != (1280, 720, 3):
        raise ValueError('idol road formation image must be canonical Maa BGR')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if after_scroll:
        # The bottom row may only have one card. Align to the full row above
        # it so its lower border cannot be mistaken for the bottom row's top.
        # Search far enough below the nominal position to cover scroll offset.
        third = _card_row_top(image, 460, through=28)
        rows = ((2, 154, third), (3, 154, third + 96))
    elif with_fourth:
        # The scrollbar reaches the top while the list can still be
        # springing back. The full first two rows move together.
        first = _card_row_top(image, 375, 29, through=42)
        rows = ((0, 29, first), (1, 154, first + 96))
    else:
        rows = tuple((row, start, top) for row, (start, top) in
                     enumerate(zip((29, 154, 154), _deck_row_tops(image), strict=True)))
    cards = []
    for row, start, y in rows:
        for column in range(6):
            x = start + 91 * column
            if not _active_formation_cell(hsv[y:y + 84, x:x + 82]):
                continue
            hue, saturation, value = hsv[y + 20:y + 44, x + 57:x + 82].transpose(2, 0, 1)
            red = ((hue <= 8) | (hue >= 172)) & (saturation >= 150) & (value >= 170)
            upgraded = bool(np.count_nonzero(red) >= 15)
            card_id = identify(image, (x, y, 82, 84), upgraded, column)
            cards.append(FormationCard(row, column, card_id, upgraded))
    return tuple(cards)


@lru_cache(maxsize=1)
def _formation_memory_icon_library(directory, size):
    items = hiro_module('vision.items')
    folder = hiro_route_file(Path('assets/pack/icons/pitem/index.json')).parent
    library = items.IconLibrary.load(folder, size)
    if not library.keys:
        return library, items.MIN_SCORE, items.MIN_MARGIN
    existing = set(library.keys)
    supplements = []
    for path in sorted((directory / 'official-art').glob('*.webp')):
        if path.stem in existing:
            continue
        pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if pixels is None or pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError('idol_road_formation_item_reference_invalid:' + path.stem)
        icon = cv2.resize(pixels, (size, size), interpolation=cv2.INTER_AREA)
        # Match extra local artwork on the same scale and mask as Hiro's PNGs.
        mask = np.repeat(icon[:, :, 3] > 128, 3).astype(np.float32)
        count = mask.sum()
        if not count:
            raise ValueError('idol_road_formation_item_reference_invalid:' + path.stem)
        # Formation tiles are bright. Render the local RGBA art on a light
        # background so transparent WebP border pixels do not dominate its edges.
        alpha = icon[:, :, 3:4].astype(np.float32) / 255
        foreground = icon[:, :, :3] * alpha + 255 * (1 - alpha)
        template = items._highpass(foreground).ravel()
        template = (template - (template * mask).sum() / count) * mask
        supplements.append((path.stem, template / (np.linalg.norm(template) + 1e-6), mask, count))
    if supplements:
        keys, templates, masks, counts = zip(*supplements, strict=True)
        library = items.IconLibrary(keys=[*library.keys, *keys],
                                    templates=np.vstack((library.templates, *templates)),
                                    masks=np.vstack((library.masks, *masks)),
                                    counts=np.concatenate((library.counts, counts)),
                                    size=size, names=library.names)
    return library, items.MIN_SCORE, items.MIN_MARGIN


def _formation_memory_item_asset(image, directory, left, top, size):
    """Run Hiro's masked icon match at the formation tile size and position."""
    try:
        library, min_score, min_margin = _formation_memory_icon_library(directory, size)
    except (ImportError, FileNotFoundError):
        return None
    crops = [image[y:y + size, x:x + size] for y in range(top - 4, top + 5, 2)
             for x in range(left - 4, left + 5, 2)]
    (score, asset), (other, _) = max(library.match_many(crops), key=lambda ranked: ranked[0][0])
    return asset if score >= min_score and score - other >= min_margin else None


def read_formation_fixed_item_assets(image, directory):
    """Identify the fixed items displayed twice on formation page one."""
    if image.shape != (1280, 720, 3):
        raise ValueError('idol_road_formation_image_not_canonical')
    size = 44
    references, _ = load_hud_references(directory, frozenset(), (size,))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    assets = []
    for slot in range(4):
        top, fixed = (283 + slot * 75, 172), (116 + slot * 70, 522)
        tx, ty = top
        fx, fy = fixed
        ribbon_sat, ribbon_value = hsv[ty:ty + size, tx:tx + size, 1:].transpose(2, 0, 1)
        tile_sat, tile_value = hsv[fy:fy + size, fx:fx + size, 1:].transpose(2, 0, 1)
        ribbon_empty = (12 <= ribbon_sat.min() <= ribbon_sat.max() <= 25
                        and 215 <= ribbon_value.min() <= ribbon_value.max() <= 232
                        and np.ptp(ribbon_sat) <= 5 and np.ptp(ribbon_value) <= 5)
        tile_empty = (tile_sat.max() <= 10 and 210 <= tile_value.min() <= tile_value.max() <= 235
                      and np.ptp(tile_value) <= 10)
        if ribbon_empty and tile_empty:
            continue
        if ribbon_empty or tile_empty:
            raise ValueError('idol_road_formation_fixed_items_unread')
        (score, asset, _), (other, _, _) = rank_icons(image, references[size], *top, size)
        (fixed_score, fixed_asset, _), (fixed_other, _, _) = rank_icons(image, references[size], *fixed, size)
        # Both renders must support the same icon. The separation from their
        # respective runner-ups is combined across the two tiles.
        if asset != fixed_asset or min(score, fixed_score) < .50 or score + fixed_score - other - fixed_other < .15:
            raise ValueError('idol_road_formation_fixed_items_unread')
        assets.append(asset)
    return tuple(assets)


def read_formation_items(image, directory, metadata, plan, *, fixed_items=None):
    """Read the fixed and memory P-item slots on the first formation page."""
    if image.shape != (1280, 720, 3):
        raise ValueError('idol_road_formation_image_not_canonical')
    metadata_by_id = {row['id']: (row['plan'], row['upgraded'] == 'TRUE') for row in metadata}
    entries = json.loads((directory / 'crosswalk.json').read_bytes())
    for row in entries:
        if 'plan' in row:
            metadata_by_id.setdefault(str(row['business_id']), (row['plan'], row['upgraded']))
    size = 44
    result = [item['id'] for item in fixed_items or ()]
    fixed_ids = frozenset(result)
    references, crosswalk = load_hud_references(directory, frozenset(metadata_by_id.keys() - fixed_ids), (size,))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # The caller supplies the fixed rows after matching the formation icons.
    # Generic icon-to-item lookup is for memory items or the older local path.
    for y in (513, 591) if fixed_items is None else (591,):
        for x in (107, 177, 247, 317):
            if y == 591:
                # An empty memory slot has a flat gray interior. Low saturation
                # alone does not prove that a faint or covered item is absent.
                saturation, value = hsv[y + 9:y + 53, x + 9:x + 53, 1:].transpose(2, 0, 1)
                if saturation.max() <= 10 and 210 <= value.min() <= value.max() <= 235 and np.ptp(value) <= 4:
                    continue
                asset = _formation_memory_item_asset(image, directory, x + 9, y + 9, size)
            else:
                cell = hsv[y + 6:y + 57, x + 6:x + 57]
                if not np.any(cell[:, :, 1] >= 100):
                    continue
                (score, best, _), (other, _, _) = rank_icons(image, references[size], x + 9, y + 9, size)
                asset = best if score >= .50 and score - other >= .15 else None
            if asset is None:
                raise ValueError(f'idol_road_formation_item_icon_unread:{x},{y}')
            hue, saturation, value = hsv[y + 3:y + 23, x + 40:x + 58].transpose(2, 0, 1)
            red = ((hue <= 8) | (hue >= 172)) & (saturation >= 150) & (value >= 170)
            upgraded = np.count_nonzero(red) >= 15
            eligible = [identifier for identifier in crosswalk.get(asset, ())
                        if identifier in metadata_by_id and metadata_by_id[identifier][0] in ('free', plan.value.lower())
                        and metadata_by_id[identifier][1] == upgraded]
            if len(eligible) != 1:
                raise ValueError(f'idol_road_formation_item_identity_unread:{asset}')
            result.append(eligible[0])
    return tuple(result)


class FormationThumbnailMatcher:
    """Use the arena's card-art evidence on a single observed formation frame."""

    def __init__(self, references, embedding, cards):
        self.references = references
        self.embedding = embedding
        self.cards = {int(row['business_id']): row for row in cards}
        self.ids_by_group = defaultdict(set)
        enabled = getattr(references, 'reference_enabled', (True,) * len(references.business_ids))
        for cid, group, active in zip(references.business_ids, references.visual_group_ids, enabled, strict=True):
            if active:
                self.ids_by_group[group].add(int(cid))

        self.scopes = {(plan, column): frozenset(cid for cid, row in self.cards.items()
                         if row['plan'] in (plan.value.lower(), 'free') and row['kind'] in kinds)
                       for plan in PlanType for column, kinds in enumerate(_SLOT_KINDS)}
        self.plan_scopes = {plan: frozenset(cid for cid, row in self.cards.items()
                                           if row['plan'] in (plan.value.lower(), 'free')) for plan in PlanType}

    def identify(self, image, box, upgraded, plan, column):
        return self._identify(image, box, self.scopes[plan, column], upgraded=upgraded)

    def identify_target(self, image, box, plan, possible):
        return self._identify(image, box, self.plan_scopes[plan] | possible, possible=possible)

    def native_card_aliases(self, missing, cards, plan, master):
        """Join gallery forms to the official IDs used by Hiro's battle catalog."""
        families = {row['internal_id'] for cid in missing if cid.isdecimal()
                    and (row := self.cards.get(int(cid))) and row.get('internal_id')}
        official = defaultdict(set)
        for row in master:
            if row['id'] in families:
                official[row['id'], row['upgradeCount']].add(row['assetId'])
        aliases = defaultdict(set)
        for cid, row in self.cards.items():
            wire, asset = row.get('internal_id'), row.get('asset_id')
            grade = int(row['upgraded'])
            key = f'{wire}+' if grade else wire
            if (wire in families and asset and official[wire, grade] == {asset}
                    and row['plan'] in (plan.value.lower(), 'free') and key in cards and str(cid) not in cards):
                aliases[key].add(str(cid))
        return {key: cid for key, ids in aliases.items() if (cid := _only(ids)) is not None}

    def _identify(self, image, box, scope, *, upgraded=None, possible=None):
        if not scope:
            return None
        reference = self.references.measure_identity(image, box, eligible_card_ids=scope)
        candidates = [row for row in reference.get('candidates', (reference,)) if row['status'] == 'MEASURED']
        groups = {row['reference_visual_group'] for row in candidates}
        if not groups:
            return None
        if len(groups) == 1 and not any(row['identity_low_confidence'] for row in candidates):
            group, = groups
        else:
            hits = self.embedding.gallery.search(self.embedding.embedder.embed(image, box), eligible_card_ids=scope)
            if not hits or hits[0].visual_group_id not in groups:
                return None
            group = hits[0].visual_group_id
        allowed = self.ids_by_group[group] & scope
        if possible is not None:
            allowed &= possible
        resolved = _only(cid for cid in allowed if upgraded is None or self.cards[cid]['upgraded'] is upgraded)
        return str(resolved) if resolved is not None else None


@lru_cache(maxsize=1)
def load_formation_thumbnail_matcher():
    base = 'resource/base/model/embedding/'
    reference = BadgeReferenceGallery.load(resolve_runtime_path(
        'assets/' + base + 'arena_badge_reference', base + 'arena_badge_reference'))
    embedding = EmbeddingCardRecognizer.load(resolve_runtime_path(
        'assets/' + base + 'arena_card', base + 'arena_card'))
    classification = resolve_runtime_path('assets/data/skill_card_classification.json',
                                          'data/skill_card_classification.json')
    if classification.is_file():
        cards = [row for row in json.loads(classification.read_text(encoding='utf-8-sig'))['cards']
                 if row['internal_id']]
    else:
        catalog = resolve_runtime_path('assets/data/skill_cards.csv', 'data/skill_cards.csv')
        with catalog.open(encoding='utf-8-sig', newline='') as stream:
            cards = [dict(business_id=int(row['id']), plan=row['plan'],
                          kind=_SOURCE_KIND.get(row['sourceType']), upgraded=row['upgraded'].upper() == 'TRUE')
                     for row in csv.DictReader(stream)]
    return FormationThumbnailMatcher(reference, embedding, cards)


def find_idol_card(idol_name, card_title, *, characters, idol_cards, allow_truncated_title=False):
    """Join the displayed character and card title to one master row."""
    if not idol_name or not card_title:
        return None
    name = ''.join(idol_name.split())
    character_id = _only(row['id'] for row in characters
                         if ''.join((row['lastName'] + row['firstName']).split()) == name)
    if character_id is None:
        return None
    cards = [row for row in idol_cards if row['characterId'] == character_id]
    exact = [row for row in cards if row['name'] == card_title]
    if exact or not allow_truncated_title:
        return exact[0] if len(exact) == 1 else None
    ellipsis = re.search(r'(?:…+|[.．]{3,})$', card_title)
    prefix = card_title[:ellipsis.start()].rstrip() if ellipsis else ''
    if not prefix:
        return None
    matches = [row for row in cards if row['name'].startswith(prefix)]
    return matches[0] if len(matches) == 1 and matches[0]['name'] != prefix else None


def resolve_formation_idol(overview, *, characters, idol_cards, initial_decks):
    """Join displayed name and title to the master, including off-page basic cards."""
    card = find_idol_card(overview.idol_name, overview.card_title, characters=characters, idol_cards=idol_cards,
                          allow_truncated_title=True)
    if card is None:
        return None
    plan = _PLANS.get(card['planType'])
    if plan is None:
        return None
    initial_cards = None
    theme = _DECK_FOR_EXAM_EFFECT.get(card['examEffectType'].removeprefix('ProduceExamEffectType_'))
    decks = [row for row in initial_decks if row['id'] == f'initial_deck-{theme}'] if theme else []
    if len(decks) == 1 and decks[0].get('produceCardIds'):
        ids = decks[0]['produceCardIds']
        upgrades = decks[0].get('produceCardUpgradeCounts') or (0,) * len(ids)
        if len(ids) == len(upgrades):
            initial_cards = tuple(zip(ids, upgrades, strict=True))
    return FormationIdol(card['id'], plan, initial_cards)
