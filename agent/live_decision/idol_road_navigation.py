"""Choose a floor from the visible Idol Road tower overview."""

import re
import json
from uuid import uuid4
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .exam_hud import battle_dialog_header


@dataclass(frozen=True)
class FloorOverview:
    number: int
    stars: int
    challenge: tuple[int, int] | None


def read_floor_overview(image, rows, challenge_box, *, idol=None):
    labels = [(''.join(row.text.split()), row.box[0] + row.box[2] // 2,
               row.box[1] + row.box[3] // 2) for row in rows]
    if not any('ステージ別ランキング' in label for label, _, _ in labels):
        return None
    if idol is not None and not any(label == ''.join(idol.split()) and x < 220 and y < 200
                                    for label, x, y in labels):
        return None
    # The stage printed in the left reward panel does not change with the carousel.
    centered = {int(label) for label, x, y in labels
                if label.isascii() and label.isdecimal() and 380 <= x <= 560 and 690 <= y <= 745}
    if len(centered) != 1:
        return None
    number = centered.pop()
    if number == 0:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    stars = 0
    for x in (382, 462, 540):
        hue, saturation, value = hsv[765:825, x - 30:x + 30].transpose(2, 0, 1)
        pink = ((hue >= 150) | (hue <= 8)) & (saturation >= 90) & (value >= 150)
        stars += int(pink.mean() >= .2)

    challenge = None
    if challenge_box is not None:
        left, top, width, height = challenge_box
        x, y = left + width // 2, top + height // 2
        hue, saturation = hsv[y - 20:y + 20, max(0, x - 100):x + 100, :2].transpose(2, 0, 1)
        orange = (hue >= 5) & (hue <= 22) & (saturation >= 150)
        if orange.mean() >= .55:
            challenge = (x, y)
    return FloorOverview(number, stars, challenge)


def read_tower_navigation(rows, idol, *, home_visible):
    """Hiro's home → contest → named idol route, using Maa OCR positions."""
    labels = [(''.join(row.text.split()).replace('ヘの道', 'への道'), row.box[0] + row.box[2] // 2,
               row.box[1] + row.box[3] // 2) for row in rows]
    if any('アイドルへの道' in text and y <= 120 for text, _, y in labels):
        progress = [(x, y) for text, x, y in labels if y > 600 and re.fullmatch(r'\d+/\d+', text)]
        names = {(text, x, y) for text, x, y in labels if len(text) >= 2 and not re.search(r'\d', text)
                 and any(abs(px - x) < 90 and 30 <= py - y <= 100 for px, py in progress)}
        key = ''.join(idol.split())
        scored = sorted((SequenceMatcher(None, text, key).ratio(), x, y) for text, x, y in names)
        if scored and scored[-1][0] >= .75 and (len(scored) == 1 or scored[-1][0] > scored[-2][0]):
            return 'idols', scored[-1][1:]
        if progress:
            return 'idol-list', tuple(sorted((text, x // 10, y // 10) for text, x, y in names))
        return None
    if (any('コンテスト' in text and 840 <= y <= 950 for text, _, y in labels)
            and any(y < 1000 and ('への道' in text or text == 'アイドル') for text, _, y in labels)):
        return 'contest', (148, 925)
    if home_visible():
        contest = {(x, y) for text, x, y in labels if text == 'コンテスト' and 430 <= x <= 570 and y > 1100}
        return 'home', contest.pop() if len(contest) == 1 else (496, 1177)
    return None


def tower_resume_button(image, ocr):
    """Hiro resumes the interrupted stage only from its own dialog."""
    if (band := battle_dialog_header(image)) is None:
        return None
    top, bottom = band
    start = max(0, top - 4)
    titles = ocr(image, roi=(20, start, 540, min(image.shape[0], bottom + 4) - start))
    title = ''.join(''.join(row.text.split()) for row in titles)
    if not re.search(r'[へヘ]の道再[開开]', title):
        return None
    return next((tuple(row.box) for row in ocr(image, roi=(0, 1000, 720, 280))
                 if row.box[1] > 1000 and row.text.strip() == '再開する'), None)


def tower_date_button(image, ocr, *, header=battle_dialog_header):
    from .hiro_runtime import hiro_module

    if header(image) is None:
        return None
    boxes = [(row.text, row.score, (x, y, x + w, y + h))
             for row in ocr(image) for x, y, w, h in (row.box,)]
    return hiro_module('loop.date_change').date_change_button(boxes)


def next_unseen_floor(origin, inspected):
    """Search toward higher floors up to the first locked one, then look below."""
    first_locked = min((number for number, floor in inspected.items() if floor.challenge is None),
                       default=max(inspected) + 2)
    for numbers in (range(origin + 1, first_locked), range(origin - 1, 0, -1)):
        for number in numbers:
            if number not in inspected:
                return number
    return None


def tower_game_day(now=None):
    """Hiro's 05:00 JST daily reset, independent of the computer's time zone."""
    jst = timezone(timedelta(hours=9))
    current = datetime.now(jst) if now is None else now.astimezone(jst)
    return (current - timedelta(hours=5)).date().isoformat()


# The large floor number below the result heading (Hiro's result LAYER_BOX).
TOWER_RESULT_LAYER_ROI = (280, 385, 160, 85)


def read_tower_clear_title(rows):
    titles = set()
    for row in rows:
        x, y, w, h = row.box
        x, y = x + w / 2, y + h / 2
        text = ''.join(row.text.split()).upper()
        if 180 <= x <= 540 and 150 <= y <= 320 and text in ('CLEAR', 'PERFECT'):
            titles.add(text)
    return next(iter(titles)) if len(titles) == 1 else None


def read_tower_clear_result(image, rows, *, ocr, layer=None):
    """Bind the result title and floor before interpreting the stars."""
    if getattr(image, 'shape', ()) != (1280, 720, 3) or (title := read_tower_clear_title(rows)) is None:
        return None
    # Read the complete number in one crop. Full-page detection can split its
    # digits into separate boxes or return them out of reading order.
    floor_pattern = r'[1-9][0-9]?'
    number_rows = tuple(ocr(image, roi=TOWER_RESULT_LAYER_ROI, expected=floor_pattern, only_rec=True))
    if len(number_rows) != 1:
        return None
    number = number_rows[0].text.strip()
    if not re.fullmatch(floor_pattern, number) or (layer is not None and int(number) != layer):
        return None
    # Hiro identifies PERFECT as the three-star result. Stars can still be
    # animating when that authoritative title has already appeared.
    if title == 'PERFECT':
        return int(number), 3
    # Sample the inside of each badge. A wide window shifted toward the center
    # also picks up the neighboring pink badge and the artwork around the gray one.
    hsv = cv2.cvtColor(image[550:615], cv2.COLOR_BGR2HSV)
    stars = 0
    for x in (210, 360, 510):
        hue, saturation, value = hsv[:, x - 36:x + 36].transpose(2, 0, 1)
        pink = ((hue >= 150) | (hue <= 8)) & (saturation >= 80) & (value >= 150)
        stars += int(pink.mean() >= .2)
    return (int(number), stars) if stars in (1, 2) else None


def read_tower_clear_stars(image, rows, *, ocr, layer=None):
    result = read_tower_clear_result(image, rows, ocr=ocr, layer=layer)
    return result[1] if result is not None else None


def tower_retry_dialog_roi(image):
    """Locate the orange header and the attached white body before reading it."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return None
    header = battle_dialog_header(image)
    if header is None:
        return None
    top, band_end = header
    # The right inner edge remains uncovered by the text and footer buttons.
    edge = image[band_end:, 665:685]
    white = ((edge.min(axis=2) >= 226) & (np.ptp(edge, axis=2) <= 32)).mean(axis=1) >= .8
    starts = np.flatnonzero(white[:24])
    if not starts.size:
        return None
    start = int(starts[0])
    ends = np.flatnonzero(~white[start:])
    if not ends.size or not 140 <= int(ends[0]) <= 450:
        return None
    bottom = band_end + start + int(ends[0])
    return 27, top, 666, bottom - top


@dataclass(frozen=True)
class TowerLoadoutCache:
    """Hiro's saved formation, bound to the game instance showing the result."""

    directory: Path
    game_created_at: int

    @property
    def path(self):
        return self.directory / f'loadout-{self.game_created_at}.json'

    def save(self, loadout, timeline, item_ids, stage):
        from route.ownership import write_hif_record

        write_hif_record(self.path, dict(version=2, loadout=asdict(loadout),
            timeline=[asdict(row) for row in timeline.values()],
            items=list(item_ids), stage=stage.identity if stage else None))

    @staticmethod
    def _opening_key(frame):
        return [frame.turns, frame.attribute, frame.score, frame.stamina, frame.genki,
                list(frame.hand), [list(row) for row in frame.drinks], list(frame.playable),
                None if frame.available_drinks is None else list(frame.available_drinks),
                [list(row) for row in frame.statuses],
                [[slot.slot, slot.kind, slot.effect_type, slot.card_id, slot.value, slot.unit, slot.turns,
                  list(slot.alternatives), slot.debuff] for slot in frame.status_slots]]

    def opening_matches(self, frame):
        return json.loads(self.path.read_bytes()).get('opening') == self._opening_key(frame)

    def has_opening(self):
        return self.path.is_file() and json.loads(self.path.read_bytes()).get('opening') is not None

    def has_verified_formation(self):
        return self.path.is_file() and json.loads(self.path.read_bytes()).get('version') == 2

    def committed_position(self):
        if self.path.is_file():
            raw = json.loads(self.path.read_bytes())
            if raw.get('active'):
                return raw.get('committed_position')
        return None

    def record_position(self, position):
        from route.ownership import write_hif_record, locked_hif_record

        with locked_hif_record(self.path):
            raw = json.loads(self.path.read_bytes())
            if position is None:
                if raw.pop('committed_position', None) is None:
                    return
            elif raw.get('active'):
                raw['committed_position'] = position
            else:
                return
            write_hif_record(self.path, raw)

    def active_id(self):
        if self.path.is_file():
            return json.loads(self.path.read_bytes()).get('active')
        return None

    def active_loadout(self):
        if self.active_id() is not None:
            return self.load()[0]
        return None

    def update_opening(self, frame):
        from route.ownership import write_hif_record, locked_hif_record

        with locked_hif_record(self.path):
            raw = json.loads(self.path.read_bytes())
            if frame is None:
                if raw.pop('opening', None) is None:
                    return
            else:
                raw.pop('committed_position', None)
                raw['opening'] = self._opening_key(frame)
                raw['active'] = uuid4().hex
            write_hif_record(self.path, raw)

    def finish(self):
        from route.ownership import write_hif_record, locked_hif_record

        with locked_hif_record(self.path):
            raw = json.loads(self.path.read_bytes())
            if raw.pop('active', None) is not None:
                raw.pop('opening', None)
                raw.pop('committed_position', None)
                write_hif_record(self.path, raw)

    def load(self, *, require_verified=False):
        from .idol_road_battle import IdolRoadLoadout
        from .idol_road_timeline import TowerTurnHint
        from .idol_road_formation import FormationCard, FormationIdol, FormationOverview
        from .high_score.contracts import PlanType

        if not self.path.is_file():
            raise ValueError('idol_road_retry_saved_formation_missing')
        raw = json.loads(self.path.read_bytes())
        if raw['version'] not in (1, 2) or (require_verified and raw['version'] != 2):
            raise ValueError('idol_road_retry_saved_formation_version')
        loadout = raw['loadout']
        overview = loadout['overview']
        for key in ('multiplier_permils', 'next_page', 'previous_page'):
            if overview[key] is not None:
                overview[key] = tuple(overview[key])
        if overview['recommended_plan'] is not None:
            overview['recommended_plan'] = PlanType(overview['recommended_plan'])
        idol = loadout['idol']
        idol['plan'] = PlanType(idol['plan'])
        if idol['initial_cards'] is not None:
            idol['initial_cards'] = tuple(tuple(card) for card in idol['initial_cards'])
        formation = IdolRoadLoadout(FormationOverview(**overview), FormationIdol(**idol),
                                    tuple(FormationCard(**row) for row in loadout['cards']))
        timeline = {row['turn']: TowerTurnHint(**row) for row in raw['timeline']}
        return formation, timeline, tuple(raw['items']), raw['stage']


@dataclass(frozen=True)
class TowerAttempts:
    directory: Path
    limit: int
    target_stars: int = 3

    def _path(self, formation):
        icon = re.fullmatch(r'i_card-([a-z0-9]+)-\d-\d+', formation.idol.idol_card_id)
        layer = formation.overview.layer
        if icon is None or type(layer) is not int or layer < 1:
            raise ValueError('idol_road_attempt_identity_unread')
        return self.stage_path(icon[1], layer)

    def stage_path(self, idol_code, layer):
        return self.directory / f'{tower_game_day()}-tower_001-{idol_code}-{layer}.json'

    @staticmethod
    def _results(path):
        if not path.exists():
            return []
        return json.loads(path.read_bytes())['results']

    def allowed(self, formation):
        return len(self._results(self._path(formation))) < self.limit

    def floor_allowed(self, idol_code, layer):
        return len(self._results(self.stage_path(idol_code, layer))) < self.limit

    def record(self, formation, result):
        from route.ownership import write_hif_record, locked_hif_record

        path = self._path(formation)
        with locked_hif_record(path):
            results = self._results(path)
            results.append(dict(rank=result.rank, score=result.score))
            write_hif_record(path, dict(results=results))
        return len(results)

    def hiro_result(self, formation, result, *, stars, receipt=None):
        """Feed persisted game results and the new result into Hiro's TowerRun."""
        return self.stage_result(self._path(formation), formation.overview.layer, result, stars=stars, receipt=receipt)

    def stage_result(self, path, layer, result, *, stars, receipt=None):
        from route.ownership import write_hif_record, locked_hif_record

        from .hiro_runtime import hiro_module

        run = hiro_module('runtime.tower').TowerRun(tower_id=path.stem[11:].rsplit('-', 1)[0])
        with locked_hif_record(path):
            results = self._results(path)
            matched = next((index for index, trial in enumerate(results)
                            if receipt is not None and trial.get('receipt') == receipt), None)
            prior = results[:matched]
            run.tries[layer] = len(prior)
            if any(trial['rank'] in (1, 2, 3) or trial.get('stars', 0) > 0 for trial in prior):
                run.cleared.add(layer)
            observed_stars = [trial['stars'] for trial in prior if 'stars' in trial]
            if observed_stars:
                run.best[layer] = max(observed_stars)
            if matched is not None:
                # The task can resume on the same result with a different goal.
                recorded = results[matched]
                stars = recorded.get('stars', 0)
                return self.verdict(run, layer, stars > 0 or recorded['rank'] in (1, 2, 3), stars)
            cleared = not getattr(result, 'failed', False) and (stars > 0 or result.rank in (1, 2, 3))
            verdict = self.verdict(run, layer, cleared, stars)
            entry = dict(rank=result.rank, score=result.score, stars=stars)
            if receipt is not None:
                entry.update(receipt=receipt, verdict=verdict)
            results.append(entry)
            write_hif_record(path, dict(results=results))
        return verdict

    def verdict(self, run, layer, cleared, stars):
        run.after_result(layer, cleared=cleared, stars=stars)
        if cleared and stars >= self.target_stars:
            return 'next'
        # Maa exposes the retry limit and star target as task settings.
        if run.tries[layer] < self.limit:
            if layer in run.gave_up_three:
                run.gave_up_three.remove(layer)
            return 'retry'
        return 'next' if run.best[layer] >= self.target_stars else 'stop'
