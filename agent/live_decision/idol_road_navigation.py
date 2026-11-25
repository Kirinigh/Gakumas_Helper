"""Choose a floor from the visible Idol Road tower overview."""

import re
import json
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import asdict, dataclass

import cv2


@dataclass(frozen=True)
class FloorOverview:
    number: int
    stars: int
    challenge: tuple[int, int] | None


def read_floor_overview(image, rows, challenge_box):
    labels = [(''.join(row.text.split()), row.box[0] + row.box[2] // 2,
               row.box[1] + row.box[3] // 2) for row in rows]
    if not any('ステージ別ランキング' in label for label, _, _ in labels):
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


def read_tower_navigation(rows, idol):
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
    if any(text == 'プロデュース' and 920 <= y <= 1070 for text, _, y in labels):
        contest = next(((x, y) for text, x, y in labels if text == 'コンテスト' and y > 1100), None)
        if contest is not None or any(text == 'ホーム' and y > 1100 for text, _, y in labels):
            return 'home', contest or (496, 1177)
    return None


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


def read_tower_clear_stars(image):
    """Use Hiro's pink star regions on the CLEAR page."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return None
    hsv = cv2.cvtColor(image[550:660], cv2.COLOR_BGR2HSV)
    stars = 0
    for x in (250, 360, 470):
        hue, saturation, value = hsv[:, x - 45:x + 45].transpose(2, 0, 1)
        pink = ((hue >= 150) | (hue <= 8)) & (saturation >= 80) & (value >= 150)
        stars += int(pink.mean() >= .2)
    return stars


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

        write_hif_record(self.path, dict(version=1, loadout=asdict(loadout),
            timeline=[asdict(row) for row in timeline.values()],
            items=list(item_ids), stage=stage.identity if stage else None))

    def load(self):
        from .idol_road_battle import IdolRoadLoadout
        from .idol_road_timeline import TowerTurnHint
        from .idol_road_formation import FormationCard, FormationIdol, FormationOverview
        from .high_score.contracts import PlanType

        if not self.path.is_file():
            raise ValueError('idol_road_retry_saved_formation_missing')
        raw = json.loads(self.path.read_bytes())
        if raw['version'] != 1:
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

    def _path(self, formation):
        icon = re.fullmatch(r'i_card-([a-z0-9]+)-\d-\d+', formation.idol.idol_card_id)
        layer = formation.overview.layer
        if icon is None or type(layer) is not int or layer < 1:
            raise ValueError('idol_road_attempt_identity_unread')
        day = tower_game_day()
        return self.directory / f'{day}-tower_001-{icon[1]}-{layer}.json'

    @staticmethod
    def _results(path):
        if not path.exists():
            return []
        return json.loads(path.read_bytes())['results']

    def allowed(self, formation):
        return len(self._results(self._path(formation))) < self.limit

    def record(self, formation, result):
        from route.ownership import write_hif_record, locked_hif_record

        path = self._path(formation)
        with locked_hif_record(path):
            results = self._results(path)
            results.append(dict(rank=result.rank, score=result.score))
            write_hif_record(path, dict(results=results))
        return len(results)

    def hiro_result(self, formation, result, *, stars):
        """Feed persisted game results and the new result into Hiro's TowerRun."""
        from route.ownership import write_hif_record, locked_hif_record

        from .hiro_runtime import hiro_module

        path = self._path(formation)
        layer = formation.overview.layer
        run = hiro_module('runtime.tower').TowerRun(tower_id=path.stem[11:].rsplit('-', 1)[0])
        with locked_hif_record(path):
            results = self._results(path)
            run.tries[layer] = len(results)
            if any(trial['rank'] <= 3 for trial in results):
                run.cleared.add(layer)
            observed_stars = [trial['stars'] for trial in results if 'stars' in trial]
            if observed_stars:
                run.best[layer] = max(observed_stars)
            verdict = run.after_result(layer, cleared=result.rank <= 3, stars=stars)
            results.append(dict(rank=result.rank, score=result.score, stars=stars))
            write_hif_record(path, dict(results=results))
        return verdict
