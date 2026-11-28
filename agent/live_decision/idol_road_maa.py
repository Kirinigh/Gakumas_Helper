"""Maa's Idol Road battle callback, from formation through the ranking page."""
from __future__ import annotations

import re
import csv
import json
import time
from types import SimpleNamespace
from pathlib import Path
from collections import Counter
from dataclasses import replace, dataclass

import cv2
import numpy as np

from agent.utils.runtime_paths import resolve_runtime_path
from agent.card_selection.model import ModelIntegrityError
from agent.card_selection.types import CandidateBox
from agent.card_selection.embedding import EmbeddingCardRecognizer, resolve_card_identity

from .upgrade import compact_ocr_text, fold_card_ocr_title, normalize_upgrade_title, detect_live_upgrade_marker
from .exam_hud import EXAM_STAMINA_ROI, valid_stamina, outlined_number_roi, battle_choice_buttons
from .exam_hud import battle_dialog_header as _battle_dialog_header
from .item_hud import read_bound_item_uses
from .exam_hand import hand_row_ready, anchor_exam_hand, read_hiro_hand_slot, read_hiro_upgrade_suffix
from .exam_loop import (
    EXAM_STALL_TIMEOUT,
    ExamResult,
    ExamFeedback,
    ExamRecovery,
    ExamNoProgress,
    ExamReobserved,
    ExamCardConfirmation,
    run_exam,
    select_exam_card,
    exam_drink_open_probe,
)
from .inventory import _inventory_ocr
from .drink_icons import EVENT_GREEN_JUICE_ID, load_default_drink_icons
from .hiro_runtime import hiro_module, hiro_ui_templates
from .hif_card_retry import read_exam_hand_strip, same_readable_resources, read_exam_select_markers, recheck_persistent_select
from .hif_entry_round import read_entry_turn
from .hif_upstream_ui import HIF_SKIP_RECOGNITION, upstream_drink_actions
from .hif_choice_panel import hif_choice_grid, _choice_selected_slot
from .hif_gkms_runtime import require_resources, battle_identity_master, require_uinet21_resources
from .idol_road_battle import (
    _wire_card,
    open_idol_road_battle,
    read_idol_road_loadout,
    read_idol_road_live_ranking,
    read_idol_road_result_frame,
)
from .idol_road_digits import read_ranking_digits
from .idol_road_status import read_hiro_statuses, hiro_primary_statuses, read_tower_status_values
from .high_score.kernel import KernelError, PythonNiaProKernel
from .idol_road_runtime import (
    ATTRIBUTES,
    TowerFrame,
    tower_legal,
    tower_opening,
    tower_branches,
    tower_live_step,
    tower_choice_step,
    tower_terminal_step,
    tower_observed_extra_action,
)
from .idol_road_timeline import tower_field_rule, complete_formation_effects, read_formation_effects_page
from .hif_capture_adapter import adapt_hif_context
from .idol_road_formation import (
    read_formation_items,
    formation_deck_at_edge,
    read_formation_overview,
    has_fourth_formation_memory,
    load_formation_thumbnail_matcher,
    read_formation_fixed_item_assets,
)
from .high_score.contracts import PlanType, ActionKind, DecisionKind, HighScoreContractError
from .idol_road_card_rules import (
    TowerCardKernel,
    bind_tower_stage,
    tower_stage_rules,
    tower_item_catalog,
    tower_skip_recovery,
    has_tower_card_rules,
    stage_rule_supported,
    tower_card_definitions,
    tower_stage_descriptions,
)
from .idol_road_navigation import (
    next_unseen_floor,
    tower_date_button,
    read_floor_overview,
    read_tower_navigation,
    tower_retry_dialog_roi,
    read_tower_clear_result,
)
from .upstream_hif_card_ui import BattleRecognitionMixin
from .high_score.catalog_v2 import compile_item_catalog, compile_drink_catalog, compile_skill_card_catalog
from .high_score.decision_box import open_native_decision_box
from .high_score.policy_factory import supports_native_policy
from .high_score.gkms_master_forms import PLAN_CODES


def _data(name):
    return resolve_runtime_path(Path('assets/data') / name, Path('data') / name)


def read_exam_card_catalog(plan, *, native):
    path = _data('skill_cards.csv')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = tuple(csv.DictReader(stream))
    free = tuple(row['id'] for row in rows if row['plan'] == 'free')
    cards = compile_skill_card_catalog(path, plan_type=plan, include_free_card_ids=free).cards
    if native:
        from .high_score.gkms_master_forms import native_card_metadata

        # Native tracks effects and fees. Reuse the same metadata for visible
        # cards even when a legacy simulator has a different or missing rule.
        tower_cards = {cid: card for cid, card in tower_card_definitions().items() if card.plan is plan}
        cards.update(tower_cards)
        metadata = native_card_metadata(rows, plan, include_master_only=True)
        cards.update((cid, card) for cid, card in metadata.items() if cid.isdecimal() and cid not in tower_cards)
        # Keep existing IDs, including bindings supplied by the legacy card map.
        binding = SimpleNamespace(cards=cards)
        forms = set()
        for cid, card in cards.items():
            try:
                forms.add(_wire_card(binding, cid, card.name.endswith('+'), native=True))
            except ValueError:
                continue
        cards.update((cid, card) for cid, card in metadata.items()
                     if cid.startswith('p_card-') and (cid.removesuffix('+'), int(cid.endswith('+'))) not in forms)
    return cards


def _center(box):
    x, y, width, height = box
    return x + width // 2, y + height // 2


_RANKING_UPDATE_STABLE_S = .6
_RANKING_UPDATE_RETRY_S = 1.5


def _same_ranking_points(first, second):
    # Match the large total in the popup, allowing minor background animation.
    return cv2.absdiff(first, second).mean() <= 6


@dataclass(frozen=True)
class TowerScreen:
    observation: TowerFrame
    boxes: tuple
    skip: tuple | None
    drinks: dict


@dataclass(frozen=True)
class TowerPreparation:
    loadout: object
    kernel: object
    deck_ids: tuple
    multipliers: tuple
    art: object
    drink_reader: object
    item_catalog: object
    item_metadata: tuple
    rust_stage: dict | None
    algorithm: str
    item_uses: dict
    timeline: dict
    native_rules: object | None = None
    unread_formation_cards: int = 0


_CHOICE_HEADERS = {
    ActionKind.HOLD_CARD: r'保留|保留するスキルカードを選択',
    ActionKind.MOVE_CARD: r'移動|移動するスキルカードを選択',
    ActionKind.FREE_PLAY: r'発動|使用|発動するスキルカードを選択|使用するスキルカードを選択',
    ActionKind.PAID_PLAY: r'使用|使用するスキルカードを選択',
}
_CHOICE_CONFIRMS = {
    ActionKind.HOLD_CARD: r'保留|決定',
    ActionKind.MOVE_CARD: r'移動|決定',
    ActionKind.FREE_PLAY: r'発動|使用|使う|決定',
    ActionKind.PAID_PLAY: r'使用|使う|決定',
}

_NATIVE_TARGET_EFFECTS = {
    11: (r'強化|強化するスキルカードを選択', r'強化|強化する|決定'),
    21: (r'コピー|作成|コピーするスキルカードを選択', r'コピー|作成|決定'),
    30: (r'コピー|複製|コピーするスキルカードを選択', r'コピー|複製|決定'),
    156: (r'成長|成長させるスキルカードを選択', r'成長|成長させる|決定'),
}
_BATTLE_CONFIRMATIONS = {
    ActionKind.END_TURN: (r'ターン終了', r'はい'),
    ActionKind.USE_DRINK: (r'使用確認', r'決定'),
    ActionKind.PLAY_CARD: (r'スキルカード使用(?:の)?確認|使用確認', r'決定|使用する'),
}


class TowerDateChanged(RuntimeError):
    """The game interrupted an active Tower run with a return to title."""


class IdolRoadMaaBattle(BattleRecognitionMixin):
    """One native Maa task owns both the formation input and the battle."""

    def __init__(self, context, verify_input, *, log):
        self.base = context
        self.verify_input = verify_input
        self.context = adapt_hif_context(context, input_guard=lambda: self._authorized())
        self.log = log
        self.sequence = 0
        self.retry_available = False
        self.rust_stage = None
        self.choice_boxes = ()
        self.choice_is_hand = False
        self.loadout = None
        self.live_screen = None
        self.observed_types = []
        self.timeline = {}
        self.item_uses = {}
        self.native_rules = None
        self.native_session = None
        self.unread_formation_cards = 0
        self.card_confirmation = None
        self.exam_recovery = ExamRecovery()
        self.unplayable_cards = set()
        self.unplayable_drinks = set()
        self._ocr_regions = {}
        self._live_reads = {}

    def _authorized(self):
        return not self.context.tasker.stopping and self.verify_input(self.context.tasker.controller) is True

    def capture(self):
        job = self.context.tasker.controller.post_screencap().wait()
        image = job.get() if job.succeeded else None
        if image is None:
            raise ValueError('idol_road_maa_capture_failed')
        self.sequence += 1
        self.image = image
        self._live_reads.clear()
        self.check_tower_date(image)
        return image

    @property
    def frame_id(self):
        return f'tower-frame-{self.sequence}'

    def click(self, point, *, purpose):
        opening_cache = getattr(self, '_tower_opening_cache', None)
        if opening_cache is not None:
            opening_cache.update_opening(None)
            self._tower_opening_cache = None
        cache = getattr(self, '_tower_result_cache', None)
        if cache is not None and getattr(self, '_saved_exam_frame', None) is not None:
            cache.record_position(None)
            self._saved_exam_frame = None
        job = self.context.tasker.controller.post_click(*point).wait()
        self.log(dict(event='idol_road_maa_input', purpose=purpose, frame_id=self.frame_id,
                      job_id=job.job_id, succeeded=job.succeeded))
        if not job.succeeded:
            raise ValueError('idol_road_maa_click_failed:' + purpose)
        return job.job_id

    def recognition(self, node, image, override=None):
        kwargs = {} if override is None else {'pipeline_override': {node: override}}
        result = self.context.run_recognition(node, image, **kwargs)
        return result if result and result.hit else None

    def target(self, node, image, override=None):
        result = self.recognition(node, image, override)
        return tuple(result.best_result.box) if result else None

    def poll(self, read, purpose, *, duration=15.):
        # The base Maa battle uses a 15 second no-progress wait. Each real
        # action or page transition has its own wait, with no battle time cap.
        limit = time.monotonic() + duration
        while not self.context.tasker.stopping:
            image = self.capture()
            if getattr(self, 'hiro_policy_route', False):
                from .hiro_maa_exam import recover_communication

                if recover_communication(self, image):
                    limit = time.monotonic() + duration
                    continue
            observed = read(image)
            if observed is not None:
                return observed
            if time.monotonic() >= limit:
                break
            time.sleep(.1)
        raise ExamNoProgress('idol_road_no_progress:' + purpose)

    def ocr(self, image, **kwargs):
        # Hiro reuses unchanged HUD crops even while the stage behind them animates.
        roi = kwargs.get('roi')
        expected = kwargs.get('expected')
        if (not isinstance(image, np.ndarray) or not roi or kwargs.keys() - {'roi', 'expected', 'only_rec'}
                or not isinstance(expected, (str, type(None)))):
            return _inventory_ocr(self.context, image, 'IdolRoadOCR', **kwargs)
        x, y, width, height = roi
        crop = image[y:y + height, x:x + width]
        key = (tuple(roi), expected, kwargs.get('only_rec', False))
        previous = self._ocr_regions.get(key)
        if previous is not None and np.array_equal(crop, previous[0]):
            return previous[1]
        rows = _inventory_ocr(self.context, image, 'IdolRoadOCR', **kwargs)
        self._ocr_regions.pop(key, None)
        if rows:
            if len(self._ocr_regions) >= 64:
                self._ocr_regions.pop(next(iter(self._ocr_regions)))
            self._ocr_regions[key] = (crop.copy(), rows)
        return rows

    def numbers(self, image, roi, *, only_rec=False):
        rows = self.ocr(image, roi=roi, expected=r'[0-9,]+', only_rec=only_rec)
        values = {int(row.text.replace(',', '')) for row in rows if re.fullmatch(r'[\d,]+', row.text)
                  and row.text.replace(',', '').isdecimal()}
        return next(iter(values)) if len(values) == 1 else None

    def ranking_number(self, image, roi):
        if roi[2] > 100 and getattr(self, 'hiro_policy_route', False):
            x, y, width, height = roi
            try:
                result, _ = hiro_module('vision.digits').read_score(image, (x, y, x + width, y + height))
            except (ImportError, RuntimeError):
                result = None
            if result is not None:
                return result
        score = read_ranking_digits(image, roi)
        if score is not None and score >= 100_000 and roi[2] > 100:
            # The last digit of a large player score can extend beyond its
            # column. The glyph reader separates it from smaller rivals.
            x, y, width, height = roi
            extended = read_ranking_digits(image, (x, y, min(width + 8, image.shape[1] - x), height))
            if extended is not None and extended // 10 == score:
                return extended
            # A bright outline can hide the last glyph even in the wider
            # crop. In that case use the existing single-line OCR on the
            # original column and accept only the one-digit completion.
            single_line = self.numbers(image, roi, only_rec=True)
            if single_line is not None and single_line // 10 == score:
                return single_line
        return score if score is not None else self.numbers(image, roi)

    def read_stamina(self, image):
        maximum = self.loadout.overview.max_stamina
        roi = EXAM_STAMINA_ROI if maximum < 100 else (573, 212, 74, 32)
        value = self.numbers(image, roi)
        if valid_stamina(value, maximum):
            return value
        if value is not None:
            self.log(dict(event='idol_road_stamina_ocr_out_of_range', frame_id=self.frame_id,
                          value=value, maximum=maximum))
        tight = outlined_number_roi(image, roi)
        fallback = self.numbers(image, tight) if tight is not None else None
        return fallback if valid_stamina(fallback, maximum) else None

    def read_genki(self, image):
        # Hiro widens the HUD field for four digits and isolates glyphs from
        # the adjacent bar. Keep the viewport edge in the recognition crop.
        tight = outlined_number_roi(image, (641, 170, 79, 44))
        if tight is not None:
            value = self.numbers(image, (tight[0], 172, tight[2], 39), only_rec=True)
            if value is None or (tight[0] < 661 and 100 <= value < 1000):
                wide = self.numbers(image, tight)
                if wide is not None and (value is None or str(wide).startswith(str(value))):
                    return wide
            if value is not None:
                return value
        return self.numbers(image, (661, 172, 59, 39))

    def formation_ocr(self, image):
        return tuple(dict(text=row.text, box=tuple(row.box)) for row in self.ocr(image))

    def check_tower_date(self, image):
        if getattr(self, 'tower_date_watch', False) and tower_date_button(
                image, self.ocr, header=_battle_dialog_header) is not None:
            raise TowerDateChanged('idol_road_date_changed')

    def resume_tower_after_date(self, idol):
        from .hiro_maa_device import HiroMaaDevice, hiro_maa_route

        def authorize():
            if not self._authorized():
                raise ValueError('the active Maa input endpoint changed')

        self.tower_date_watch = False
        date = hiro_module('loop.date_change')
        session = SimpleNamespace(recover=lambda event, detail: self.log(dict(event='hiro_tower_' + event,
                                                                              detail=detail)))
        device = HiroMaaDevice(self.context, authorize, log=self.log)
        with hiro_maa_route(device, self.ocr, session=session) as runtime:
            try:
                hiro_module('runtime.play').recover_date_change(runtime, session)
            except date.DataLinkBlocked as error:
                raise ValueError('idol_road_data_link_required') from error
        self.enter_idol_road(idol)

    def enter_idol_road(self, idol):
        settle = hiro_module('runtime.tower_play').ACT_WAIT_S

        def read(image):
            if _battle_dialog_header(image) is not None:
                # Maa owns the resume click. Other dialogs can cover live navigation buttons.
                return ('resume', None) if self.recognition('IdolRoadResumeSaved', image) else None
            rows = self.ocr(image)
            if read_floor_overview(image, rows, None, idol=idol) is not None:
                return 'arrived', None

            def home_visible():
                with hiro_ui_templates():
                    return hiro_module('vision.classify').at_home(image)

            return read_tower_navigation(rows, idol, home_visible=home_visible)

        def advance(page, point, destinations):
            purpose = 'tower-entry-' + page
            after = 'tower-entry-after-' + page

            def destination(image):
                branch = read(image)
                return branch if branch is not None and branch[0] in destinations else None

            self.click(point, purpose=purpose)
            try:
                return self.poll(destination, after, duration=settle)
            except ExamNoProgress:
                if self.context.tasker.stopping:
                    raise
                fresh = read(self.capture())
                if fresh is not None:
                    if fresh[0] in destinations:
                        return fresh
                    if fresh[0] == page and all(abs(a - b) <= 24 for a, b in zip(point, fresh[1], strict=True)):
                        self.click(fresh[1], purpose=purpose + '-retry')
                return self.poll(destination, after)

        terminal = ('arrived', 'resume')
        page, point = self.poll(read, 'tower-entry')
        if page == 'home':
            page, point = advance(page, point, ('contest', 'idols', 'idol-list', *terminal))
        if page == 'contest':
            page, point = advance(page, point, ('idols', 'idol-list', *terminal))
        if page == 'idol-list' and idol:
            # The DMM list shows fewer rows than Hiro's capture. Scroll this
            # list and use the same name-and-progress match in each viewport.
            for start, end in ((1110, 475), (475, 1110)):
                seen = set()
                while page == 'idol-list' and point not in seen:
                    seen.add(point)
                    job = self.context.tasker.controller.post_swipe(360, start, 360, end, 500).wait()
                    self.log(dict(event='idol_road_idol_list_swipe', job_id=job.job_id, succeeded=job.succeeded))
                    if not job.succeeded:
                        raise ValueError('idol_road_idol_list_swipe_failed')
                    time.sleep(1.25)
                    page, point = self.poll(read, 'tower-idol-list-after-swipe')
                if page != 'idol-list':
                    break
        if page == 'idols':
            page, point = advance(page, point, terminal)
        if page in terminal:
            return
        raise ValueError('idol_road_entry_did_not_reach_target')

    def enter_challengeable_floor(self, tower_run=None, *, attempts=None, target_stars=3):
        tower = hiro_module('runtime.tower_play') if attempts is not None else None
        layers = hiro_module('vision.tower') if tower is not None else None
        if attempts is not None:
            target_stars = attempts.target_stars
        idol_code = None

        def read(image):
            nonlocal idol_code
            rows = self.ocr(image)
            floor = read_floor_overview(image, rows, self.target('IdolRoadStartFlag', image))
            if floor is not None and tower is not None and idol_code is None:
                boxes = [(''.join(row.text.split()), (x, y, x + w, y + h))
                         for row in rows for x, y, w, h in (row.box,)]
                idol_code = tower.idol_code(layers._layers_fields(image, boxes).get('idol'))
            return floor

        floor = self.poll(read, 'floor-overview')
        origin = floor.number
        inspected = {}
        step_distance = 170.0
        max_swipes = 30
        for moves in range(max_swipes + 1):
            while floor.challenge is None:
                # Hiro only counts a missing button twice on the same centered floor.
                next_view = self.poll(read, 'floor-challenge')
                same_floor = floor.number == next_view.number
                floor = next_view
                if same_floor:
                    break
            eligible = floor.challenge is not None and floor.stars < target_stars
            if eligible and tower_run is not None:
                eligible = tower_run.pick_layer([dict(layer=floor.number, stars=floor.stars, unlocked=True)]) is not None
            if eligible and (idol_code is None or attempts.floor_allowed(idol_code, floor.number)):
                self.click(floor.challenge, purpose=f'floor-challenge-{floor.number}')
                return
            inspected[floor.number] = floor
            target = next_unseen_floor(origin, inspected)
            if target is None or moves == max_swipes:
                break

            towards_higher = target > floor.number
            distance = int(min(520, max(80, step_distance * abs(target - floor.number))))
            y = 380 if towards_higher else 900
            end = y + distance if towards_higher else y - distance
            job = self.context.tasker.controller.post_swipe(470, y, 470, end, 500).wait()
            self.log(dict(event='idol_road_floor_swipe', current=floor.number, target=target, job_id=job.job_id,
                          succeeded=job.succeeded))
            if not job.succeeded:
                raise ValueError('idol_road_floor_swipe_failed')
            time.sleep(1.5)
            after = self.poll(read, 'floor-after-swipe')
            moved = abs(after.number - floor.number)
            measured = (step_distance + distance / moved) / 2 if moved else step_distance * 1.6
            step_distance = min(420, max(80, measured))
            floor = after
        goal = '未满三星' if target_stars == 3 else '未通关'
        raise ValueError(f'偶像之路总览未找到可挑战的{goal}楼层')

    def show_page(self, point, page):
        self.click(point, purpose=f'formation-page-{page}')

        def read(image):
            found = read_formation_overview(self.formation_ocr(image))
            return image if found and found.page == page else None

        return self.poll(read, f'formation-page-{page}')

    def scroll_formation_deck(self, toward_bottom):
        start, end = (583, 430) if toward_bottom else (430, 583)
        job = self.context.tasker.controller.post_swipe(550, start, 550, end, 350).wait()
        purpose = 'scroll-formation-deck'
        self.log(dict(event='idol_road_maa_input', purpose=purpose, frame_id=self.frame_id,
                      job_id=job.job_id, succeeded=job.succeeded))
        if not job.succeeded:
            raise ValueError('idol_road_formation_deck_scroll_failed')
        return self.poll(lambda frame: frame if formation_deck_at_edge(frame, toward_bottom) else None, purpose)

    def restore_formation_error_view(self, overview, *, row=None):
        """Leave the relevant slots visible for Maa's existing failure screenshot."""
        page = 1 if row is None else 2

        def same_floor(image):
            found = read_formation_overview(self.formation_ocr(image))
            if found is None or overview.layer is None or found.layer != overview.layer:
                raise ValueError('idol_road_formation_error_view_unverified')
            return found

        try:
            image = self.capture()
            current = same_floor(image)
            if page == 1 and current.page == 2 and current.previous_page is not None:
                image = self.show_page(current.previous_page, 1)
                current = same_floor(image)
            if current.page != page:
                raise ValueError('idol_road_formation_error_view_unverified')
            if row is not None and has_fourth_formation_memory(image):
                toward_bottom = row >= 2
                if not formation_deck_at_edge(image, toward_bottom):
                    image = self.scroll_formation_deck(toward_bottom)
                    if same_floor(image).page != page:
                        raise ValueError('idol_road_formation_error_view_unverified')
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            self.log(dict(event='idol_road_formation_error_view_unavailable', page=page, row=row, reason=str(error)))
        else:
            self.log(dict(event='idol_road_formation_error_view_restored', page=page, row=row))

    @staticmethod
    def _art():
        model = Path('resource/base/model/classify/card_embedding')
        return EmbeddingCardRecognizer.load(resolve_runtime_path(Path('assets') / model, model))

    def prepare(self, algorithm, *, cached=None, remember=None):
        self.algorithm = algorithm
        self.rust_stage = None
        hiro_tables = None
        if algorithm == 'rust-gnrpa':
            from .high_score.tower_gnrpa import load_rust_tower

            _, hiro_tables = load_rust_tower()
        elif supports_native_policy(algorithm):
            from .high_score.tower_gnrpa import load_native_tower

            if algorithm == 'uinet-1.1':
                require_resources()
            elif algorithm == 'uinet-2.1':
                require_uinet21_resources()
            _, hiro_tables = load_native_tower()
        elif algorithm != 'strong':
            raise ValueError('idol_road_unsupported_policy')
        self.art = self._art()
        native = hiro_tables is not None
        icons = _data('hif_item_icons')
        items_path = _data('p_items.csv')
        with items_path.open(encoding='utf-8-sig', newline='') as stream:
            self.item_metadata = tuple(row for row in csv.DictReader(stream) if row['mode'] == 'stage')
        stage = None
        item_ids = ()
        formation = None

        def on_overview(first_page, overview, _idol):
            nonlocal formation
            formation = first_page
            self.timeline = self.inspect_formation_effects(first_page, overview.layer)

        if cached is None:
            tables = battle_identity_master()
            initial = json.loads(_data('hif_fast/master/ExamInitialDeck.json').read_text(encoding='utf-8'))
            self.loadout = read_idol_road_loadout(self.capture(), ocr=self.formation_ocr, show_page=self.show_page,
                identify=self.formation_card, characters=tables['Character'], idol_cards=tables['IdolCard'],
                initial_decks=initial, on_overview=on_overview, scroll_deck=self.scroll_formation_deck)
            overview = self.loadout.overview
            layer = overview.layer
            stage = bind_tower_stage(layer, self.timeline) if (native or layer >= 15) and self.timeline else None
            try:
                if native or stage is not None:
                    assets = read_formation_fixed_item_assets(formation, icons)
                    if stage is None and (self.timeline or assets):
                        raise ValueError('idol_road_native_gimmick_variant_unread')
                    if stage is not None and Counter(assets) != Counter(item['assetId'] for item in stage.fixed_items):
                        raise ValueError('idol_road_formation_fixed_items_mismatch')
                fixed_items = stage.fixed_items if stage is not None else () if native else None
                item_ids = read_formation_items(formation, icons, self.item_metadata, self.loadout.idol.plan,
                                                fixed_items=fixed_items)
            except (OSError, ValueError, cv2.error):
                self.restore_formation_error_view(overview)
                raise
        else:
            self.loadout, self.timeline, item_ids, stage_id = cached
            stage = tower_stage_rules(self.loadout.overview.layer, stage_id) if stage_id is not None else None
            if native and stage is None and self.timeline:
                raise ValueError('idol_road_native_gimmick_variant_unread')
        if native:
            from .high_score.tower_gnrpa import RustTowerStageBindingMissing, resolve_rust_tower_stage

            try:
                binding = resolve_rust_tower_stage(hiro_tables, self.loadout, observed_stage=stage)
            except RustTowerStageBindingMissing as error:
                if algorithm == 'rust-gnrpa':
                    require_uinet21_resources()
                    self.algorithm = 'uinet-2.1'
                    self.log(dict(event='idol_road_policy_fallback', requested_algorithm=algorithm,
                                  algorithm=self.algorithm, reason=str(error)))
            else:
                if algorithm == 'rust-gnrpa':
                    self.rust_stage = binding
        plan = self.loadout.idol.plan
        if stage is not None:
            stage_event = dict(event='idol_road_native_stage', id=stage.identity,
                               fixed_items=[item['id'] for item in stage.fixed_items],
                               backend='hiro-rust' if native else 'python')
            if not native:
                stage_event['python_unmapped'] = [turn for turn, row in stage.rows.items()
                                                  if not stage_rule_supported(row)]
            self.log(stage_event)
        cards = read_exam_card_catalog(plan, native=native)
        unread = [card for card in self.loadout.cards if card.card_id is None]
        if unread and not native:
            if cached is None:
                self.restore_formation_error_view(self.loadout.overview, row=unread[0].row)
            raise ValueError('idol_road_formation_has_unread_or_unsupported_card')
        self.unread_formation_cards = len(unread)
        if unread:
            self.log(dict(event='idol_road_formation_unread_cards', slots=[(card.row, card.column) for card in unread]))
        formation_ids = tuple(card.card_id for card in self.loadout.cards if card.card_id is not None)
        if native and (missing := set(formation_ids) - cards.keys()):
            cards = self.bind_native_formation_catalog(cards, missing, plan)
        kernel_type = PythonNiaProKernel
        if stage is not None or has_tower_card_rules(formation_ids):
            if not native:
                cards.update((cid, card) for cid, card in tower_card_definitions().items() if card.plan is plan)
            kernel_type = TowerCardKernel
        drinks_path = _data('p_drinks.csv')
        drinks = compile_drink_catalog(drinks_path, plan_type=plan).drinks
        self.item_catalog = compile_item_catalog(items_path, plan_type=plan)
        if kernel_type is TowerCardKernel:
            self.item_catalog, self.item_metadata = tower_item_catalog(
                self.item_catalog, self.item_metadata, stage=None if native else stage, plan=plan)
        self.kernel = kernel_type(cards, drinks, self.item_catalog.items)
        if stage is not None and not native:
            self.kernel.use_native_stage(stage)
        native_items = {}
        item_names = {row['id']: row['name'] for row in self.item_metadata}
        if stage:
            item_names.update((item['id'], item['name']) for item in stage.fixed_items)
        for item_id in item_ids:
            if native:
                # The icon crosswalk can already return a Hiro master ID.
                # Local names are only needed to translate legacy numeric IDs.
                native_items[item_id] = item_names.get(item_id)
            if item_id not in self.kernel.items and not native:
                raise ValueError('idol_road_formation_item_rule_unavailable:' + item_id)
            definition = self.kernel.items.get(item_id)
            effects = definition.persistent_effects if definition else ()
            limits = {effect.schedule_limit for effect in effects}
            if len(limits) > 1:
                raise ValueError('idol_road_formation_item_use_limit_ambiguous:' + item_id)
            self.item_uses[item_id] = next(iter(limits), None)
        if native:
            from .idol_road_native import NativeTowerRules

            self.native_rules = NativeTowerRules(self.kernel, stage, items=native_items)
        self.log(dict(event='idol_road_formation_items', items=self.item_uses))
        inverse_forms = self.native_rules.cards if native else {}
        if not native:
            for cid, definition in cards.items():
                try:
                    key = _wire_card(self.kernel, cid, definition.name.endswith('+'))
                except ValueError:
                    continue
                inverse_forms.setdefault(key, []).append(cid)
        basic = self.loadout.idol.initial_cards
        if basic is None or any(len(inverse_forms.get(key, ())) != 1 for key in basic):
            raise ValueError('idol_road_basic_deck_identity_unmapped')
        self.deck_ids = formation_ids + tuple(inverse_forms[key][0] for key in basic)
        if any(cid not in cards for cid in self.deck_ids):
            failing = [card for card in self.loadout.cards if card.card_id is not None and card.card_id not in cards]
            self.log(dict(event='idol_road_formation_unsupported_cards',
                          slots=[(card.row, card.column, card.card_id) for card in failing]))
            if cached is None:
                self.restore_formation_error_view(self.loadout.overview, row=failing[0].row if failing else None)
            raise ValueError('idol_road_formation_has_unread_or_unsupported_card')
        with drinks_path.open(encoding='utf-8-sig', newline='') as stream:
            drink_ids = tuple(row['id'] for row in csv.DictReader(stream))
        self.drink_reader = load_default_drink_icons(_data('hif_drink_icons'), (*drink_ids, EVENT_GREEN_JUICE_ID))
        self.multipliers = self.loadout.overview.multiplier_permils
        if not self.multipliers or not self.loadout.overview.max_stamina or not self.loadout.overview.turns:
            raise ValueError('idol_road_formation_turns_or_resources_unread')
        if cached is None and remember is not None:
            remember.save(self.loadout, self.timeline, item_ids, stage)

    def bind_native_formation_catalog(self, cards, missing, plan):
        # Keep the numeric IDs used by the artwork readers, with exactly one
        # local definition per official form for Hiro's reverse lookup.
        master = hiro_module('bench_bridge.tower').master().rows('ProduceCard')
        aliases = load_formation_thumbnail_matcher().native_card_aliases(missing, cards, plan, master)
        verified = {}
        for official, numeric in aliases.items():
            upgraded = official.endswith('+')
            definition = replace(cards[official], card_id=numeric)
            try:
                form = _wire_card(SimpleNamespace(cards={numeric: definition}), numeric, upgraded, native=True)
            except ValueError:
                continue
            if form == (official.removesuffix('+'), int(upgraded)):
                verified[official] = numeric
        result = cards.copy()
        for cid, card in cards.items():
            if cid in verified or card.base_card_id in verified or card.upgrade_card_id in verified:
                key = verified.get(cid, cid)
                result.pop(cid)
                result[key] = replace(card, card_id=key,
                    base_card_id=verified.get(card.base_card_id, card.base_card_id),
                    upgrade_card_id=verified.get(card.upgrade_card_id, card.upgrade_card_id))
        return result

    def preparation(self):
        return TowerPreparation(self.loadout, self.kernel, self.deck_ids, self.multipliers, self.art,
            self.drink_reader, self.item_catalog, self.item_metadata, self.rust_stage, self.algorithm,
            self.item_uses, self.timeline, self.native_rules, self.unread_formation_cards)

    def enter_same_formation_retry(self):
        def dialog(image):
            roi = tower_retry_dialog_roi(image)
            if roi is None:
                return None
            x, y, w, h = roi
            rows = [(re.sub(r'\s+', '', row.text or ''), row.box) for row in self.ocr(image, roi=roi)
                    if x <= row.box[0] and row.box[0] + row.box[2] <= x + w
                    and y <= row.box[1] and row.box[1] + row.box[3] <= y + h]
            text = ''.join(value for value, _ in rows)
            if '再挑戦確認' not in text or 'ステージを再挑戦' not in text:
                return None
            if re.search(r'ジュエル|アイテム|チケット|回数', text):
                raise ValueError('idol_road_retry_dialog_shows_cost')
            same = [box for value, box in rows if '同じ編成で再挑戦' in value]
            ok = self.target('IdolRoadRetryConfirm', image)
            if len(same) != 1 or ok is None:
                return None
            ox, oy = _center(ok)
            # Hiro locates the footer independently of the dialog text.
            # Rounded corners can end the white text region above the OK center.
            return (tuple(same[0]), ok) if x <= ox < x + w and 900 <= oy <= 1260 else None

        def retry_page(image):
            found = dialog(image)
            receipt = getattr(self, '_tower_retry_result', None)
            if found is not None or receipt is None or tower_retry_dialog_roi(image) is not None:
                return found
            sent, layer, failed, stars, result = receipt
            if time.monotonic() - sent < hiro_module('runtime.tower_play').RESULT_REDO_S:
                return None
            target = self.target('IdolRoadRetry', image)
            if target is None:
                return None
            if failed:
                current = self.read_result_page(image) or self.read_unscored_failure(image)
                same = current is not None and (current['rank'], current['score']) == (result.rank, result.score)
            else:
                same = read_tower_clear_result(image, self.ocr(image), ocr=self.ocr, layer=layer) == (layer, stars)
            if same:
                self.click(_center(target), purpose='tower-result-retry-repeat')
                self._tower_retry_result = None
            return None

        label, _ = self.poll(retry_page, 'same-formation-retry-dialog')
        self._tower_retry_result = None
        self.click(_center(label), purpose='retry-same-formation')
        _, ok = self.poll(dialog, 'same-formation-retry-confirm')
        self.click(_center(ok), purpose='retry-confirm')

    def read_tower_opening(self, image):
        opening = self.observe_live(image)
        if opening is None or opening.observation.turns != self.loadout.overview.turns:
            return None
        hand = opening.observation.hand
        if hand:
            # Hiro's hand signature is absent before the A/M/T badges are visible.
            with hiro_ui_templates():
                if not hiro_module('vision.hand_slots').card_letters(image):
                    return None
        unread = self.unread_formation_cards
        if unread is not None and sum((Counter(hand) - Counter(self.deck_ids)).values()) > unread:
            return None
        return opening

    def stable_tower_opening(self):
        gap = hiro_module('exec.exam').STABLE_GAP
        previous = None

        def read(image):
            nonlocal previous
            now = time.monotonic()
            opening = self.read_tower_opening(image)
            if opening is None:
                previous = None
                return None
            frame = opening.observation
            signature = (self.read_progress_signature(image, screen=opening), frame.attribute, frame.score,
                         frame.hand, frame.playable, frame.drinks, frame.available_drinks)
            if previous is None or previous[0] != signature:
                previous = signature, now
            elif now - previous[1] >= gap:
                return opening
            return None

        return read

    def after_retry(self, image, read_opening):
        opening = read_opening(image)
        if opening is not None:
            return 'live', opening
        if self.target('IdolRoadStart', image) and read_formation_overview(self.formation_ocr(image)) is not None:
            return 'formation', None
        return None

    def formation_card(self, image, box, upgraded, plan, column):
        if not hasattr(self, 'formation_matcher'):
            self.formation_matcher = load_formation_thumbnail_matcher()
        return self.formation_matcher.identify(image, box, upgraded, plan, column)

    def choice_card_id(self, image, box, slot, possible_forms):
        card_id = self.card_id(image, box, slot)
        if card_id in possible_forms:
            return card_id
        # These lists show square thumbnails, like the formation page.
        # Match the artwork against the live effect's known card instances.
        matcher = load_formation_thumbnail_matcher()
        numeric_forms = frozenset(int(cid) for cid in possible_forms if cid.isdecimal())
        return (matcher.identify_target(image, box, self.loadout.idol.plan, numeric_forms)
                if numeric_forms else None)

    def scoped_card_id(self, image, box, candidates, plan, *, dimmed):
        """Resolve artwork within the observed deck with the normal match cutoffs."""
        candidates = frozenset(int(cid) for cid in candidates if cid.isdecimal())
        if not candidates:
            return None
        x, y, width, height = box
        crop = image[y:y + height, x:x + width]
        if dimmed:
            crop = cv2.convertScaleAbs(crop, alpha=1.85)
        local_box = (0, 0, width, height)
        gallery = self.art.gallery
        hits = gallery.search(self.art.embedder.embed(crop, local_box),
                              eligible_card_ids=candidates)
        if (len(hits) < 2 or hits[0].similarity < .70
                or hits[0].similarity - hits[1].similarity < .05):
            return None
        upgraded = gallery.infer_upgrade_state(crop, local_box, hits[0],
            source_color_order=self.art.embedder.source_color_order,
            maximum_distance=self.art.upgrade_maximum_distance, minimum_margin=self.art.upgrade_minimum_margin)
        try:
            return resolve_card_identity(hits, plan=plan, upgraded=upgraded).card_id
        except ValueError:
            return None

    def card_id(self, image, box, slot, *, visible_right=None, possible_forms=None, dimmed=False):
        primary = self._gkh_card_id(image, box, slot, visible_right=visible_right,
                                    possible_forms=possible_forms, dimmed=dimmed)
        if primary is not None and (possible_forms is None or primary in possible_forms):
            return primary
        return self.hiro_card_id(image, box, slot, possible_forms=possible_forms)

    def read_hand_strip(self, image):
        cached = getattr(self, '_hand_strip', None)
        if cached is None or cached[0] is not image:
            cached = self._hand_strip = image, read_exam_hand_strip(self.context, image)
        return cached[1]

    def live_hand_titles(self, image, boxes):
        if not boxes or not getattr(self, 'hiro_policy_route', False):
            return {}
        result = self.read_hand_strip(image)
        rows = getattr(result, 'all_results', ()) or ()  # Includes titles when the SELECT filter has no hit.
        if not rows:
            return {}
        cards = self.kernel.cards
        cached = getattr(self, '_hand_title_catalog', None)
        if cached is None or cached[0] is not cards:
            titles = {}
            for cid, card in cards.items():
                key = normalize_upgrade_title(compact_ocr_text(card.name))
                titles.setdefault(key, set()).add(cid)
            cached = self._hand_title_catalog = cards, titles
        titles = cached[1]
        identified = {}
        tracked = None
        for slot, (left, top, width, height) in enumerate(boxes):
            right = min(left + width, boxes[slot + 1][0]) if slot + 1 < len(boxes) else left + width
            clipped = right < left + width - 4
            matches = set()
            for row in rows:
                x, y, w, h = row.box
                if row.score >= .90 and left <= x < x + w <= right and top + height - 40 <= y + h / 2 <= top + height:
                    key = normalize_upgrade_title(compact_ocr_text(row.text or ''))
                    exact = titles.get(key)
                    if exact:
                        matches.update(exact)
                    elif clipped and row.score >= .95 and right - (x + w) <= 8 and len(key) >= 4:
                        # Hiro uses a prefix when the following card covers the end of the title.
                        forms = {cid for title, cids in titles.items() if title.startswith(key) for cid in cids}
                        if len({cards[cid].base_card_id or cid for cid in forms}) == 1:
                            if tracked is None:
                                session = getattr(self, 'native_session', None)
                                tracked = set(self.deck_ids)
                                if session is not None:
                                    tracked.update(session.drawable_card_forms())
                            known = forms & tracked
                            if len(known) == 1:
                                matches.update(known)
            if len(matches) == 1:
                identified[slot], = matches
        return identified

    def prefer_live_hand_title(self, artwork, title):
        if title is None or artwork is None or title == artwork:
            return title or artwork
        a, b = self.kernel.cards[artwork], self.kernel.cards[title]
        # A clipped suffix cannot downgrade a form independently supported by the artwork reader.
        if (a.base_card_id or artwork) == (b.base_card_id or title) and a.name.count('+') > b.name.count('+'):
            return artwork
        return title

    def hiro_card_id(self, image, box, slot, *, possible_forms=None):
        row = read_hiro_hand_slot(image, box, slot, self.ocr)
        if not row or not row.get('card_id'):
            return None
        identifier, grade = row['card_id'], row['upgrade']
        if row.get('plus') != grade:
            return None
        forms = getattr(getattr(self, 'native_rules', None), 'cards', None)
        if forms is None:
            # The Python policy uses the same master IDs but has no native inverse index.
            forms = {}
            for cid, card in self.kernel.cards.items():
                if card.name.rstrip('+') != row['name'].rstrip('+'):
                    continue
                try:
                    form = _wire_card(self.kernel, cid, card.name.endswith('+'), native=True)
                except ValueError:
                    continue
                forms.setdefault(form, []).append(cid)
        if grade is None and possible_forms is None:
            return None
        candidates = {cid for (wire, count), local in forms.items() if wire == identifier
                      and (grade is None or grade == count) for cid in local}
        if possible_forms is not None:
            candidates.intersection_update(possible_forms)
        elif len(candidates) > 1:
            session = self.native_session
            known = session.drawable_card_forms() if session is not None else self.deck_ids
            candidates.intersection_update(known)
        return next(iter(candidates)) if len(candidates) == 1 else None

    def fill_hiro_hand_slots(self, image, raw, controls):
        """Supply GKH artwork evidence for unread slots in Hiro's screen-only route."""
        hand = raw.get('hand') or ()
        if (raw.get('hand_layout') != 'ok' or len(hand) != len(controls)
                or not any(not row.get('card_id') for row in hand)):
            return raw
        if any(row.get('slot') != slot or control is None or (row.get('box') and (
            abs(control.rect.x1 - row['box'][0]) > 26 or abs(control.rect.y1 - row['box'][1]) > 24))
            for slot, (row, control) in enumerate(zip(hand, controls, strict=True))):
            return raw
        plan_types = {f'ProducePlanType_Plan{code - 1}': PlanType(name) for name, code in PLAN_CODES.items()}
        visible = {plan_types[row['plan']] for row in hand if row.get('plan') in plan_types}
        if self.loadout is not None:
            visible.add(self.loadout.idol.plan)
        if len(visible) > 1:
            return raw
        art = getattr(self, 'art', None) or getattr(self, '_screen_hand_art', None)
        if art is None:
            try:
                art = self._screen_hand_art = self._art()
            except (ImportError, OSError, ModelIntegrityError):
                return raw
        catalogs = getattr(self, '_screen_hand_catalogs', None)
        if catalogs is None:
            catalogs = self._screen_hand_catalogs = {}
        cards = hiro_module('vision.cards')
        result = list(hand)
        for slot, (row, control) in enumerate(zip(hand, controls, strict=True)):
            if row.get('card_id'):
                continue
            letter = control.letter
            left, top = letter.center.x - 96, letter.y1 - 200
            next_left = controls[slot + 1].letter.center.x - 96 if slot + 1 < len(controls) else image.shape[1]
            width = min(192, image.shape[1] - left)
            if left < 0 or top < 0 or width < 1:
                continue
            forms = set()
            for plan in visible or PlanType:
                if plan not in catalogs:
                    try:
                        catalogs[plan] = SimpleNamespace(cards=read_exam_card_catalog(plan, native=True))
                    except (ImportError, OSError, ValueError):
                        return raw
                kernel = catalogs[plan]
                cid = self._gkh_card_id(image, (left, top, width, 248), slot,
                    visible_right=next_left if next_left < left + width - 4 else None,
                    dimmed=not row.get('available'), screen_catalog=(plan, kernel.cards, art))
                definition = kernel.cards.get(cid)
                if definition is not None:
                    try:
                        forms.add(_wire_card(kernel, definition.card_id, definition.name.endswith('+'), native=True))
                    except ValueError:
                        continue
            if len(forms) == 1:
                version = cards.card_version(*next(iter(forms)))
                if version is not None:
                    result[slot] = {**row, **cards.card_summary(version), 'source': 'gkh-hand'}
        if all(new is old for new, old in zip(result, hand, strict=True)):
            return raw
        return {**raw, 'hand': result, 'unrecognized_cards': sum(not row.get('card_id') for row in result)}

    def _gkh_card_id(self, image, box, slot, *, visible_right=None, possible_forms=None, dimmed=False,
                     screen_catalog=None):
        if screen_catalog is None:
            plan, cards, art = self.loadout.idol.plan, self.kernel.cards, self.art
            deck_ids, session = self.deck_ids, getattr(self, 'native_session', None)
        else:
            plan, cards, art = screen_catalog
            deck_ids, session = (), None
        plan = plan.value.lower()
        x, y, width, height = box
        right = min(visible_right, x + width - 4) if visible_right is not None else x + width - 4
        title_roi = (x + 4, y + height - 36, right - x - 4, 36)
        covered_center = width // 2 - 4 if visible_right is not None and visible_right < x + width - 4 else None

        def read_form(card_id):
            # Hiro reads the visible title suffix separately from the artwork.
            definition = cards.get(card_id)
            upgrade = getattr(definition, 'upgrade_card_id', None)
            if (upgrade in cards and isinstance(image, np.ndarray)
                    and title_roi[2] > 0 and (possible_forms is None or upgrade in possible_forms)
                    and (detect_live_upgrade_marker(image, slot, box=title_roi).upgraded
                         or read_hiro_upgrade_suffix(image, title_roi, center_x=covered_center))):
                return upgrade
            return card_id

        prediction = art.classify(image, CandidateBox(slot, tuple(box)), frame_id=self.frame_id,
            plan=plan, acceptance_threshold=.70, min_margin=.05)
        if prediction.accepted:
            card_id = read_form(prediction.card_id)
            if (not deck_ids and card_id in cards
                    or prediction.card_id in deck_ids or card_id in deck_ids
                    or (possible_forms is not None and card_id in possible_forms)) and (
                    possible_forms is None or card_id in possible_forms):
                return card_id
        matched = None
        if isinstance(image, np.ndarray):
            known = possible_forms if possible_forms is not None else deck_ids
            matched = self.scoped_card_id(image, box, known, plan, dimmed=dimmed)
            # A close global art tie still needs Hiro's title cross-check,
            # even when the smaller deck slice returns a single hit.
            if matched and (visible_right is not None or getattr(prediction, 'reason', '') != 'below_margin_threshold'):
                return read_form(matched)

        def normalized(text):
            # Hiro normalizes width before comparing OCR text to master names.
            return normalize_upgrade_title(compact_ocr_text(text).strip('[]'))

        rows = self.ocr(image, roi=title_roi) if title_roi[2] > 0 else ()
        titles = {normalized(row.text) for row in rows if row.text}
        if len(rows) > 1:
            # Hiro combines the fragments of one card's cropped name strip.
            titles.add(normalized(''.join(row.text for row in sorted(rows, key=lambda row: row.box[0]))))
        if visible_right is not None:
            def clipped_matches(known):
                names = {cid: normalized(cards[cid].name) for cid in known}
                marked = {cid for cid, name in names.items() if name.endswith('+') and name in titles}
                if marked:
                    return marked
                if possible_forms is None:
                    names = {cid: name.rstrip('+') for cid, name in names.items()}
                found_cards = set()
                for title in titles:
                    # Hiro's clipped-name fallback trims up to two misread trailing glyphs.
                    for end in range(len(title), max(3, len(title) - 2) - 1, -1):
                        prefix = fold_card_ocr_title(title[:end])
                        matches = {cid for cid, name in names.items() if fold_card_ocr_title(name).startswith(prefix)}
                        if matches:
                            found_cards.update(matches)
                            break
                    found_cards.update(cid for cid, name in names.items() if name == title)
                return found_cards

            candidates = set()
            if titles and possible_forms is None and session is not None:
                # The live zones can include a form learned or changed since formation.
                candidates = clipped_matches(session.drawable_card_forms())
            if not candidates:
                known = possible_forms if possible_forms is not None else deck_ids
                candidates = clipped_matches(known)
            if len(candidates) == 1:
                return next(iter(candidates))
            if candidates or possible_forms is not None:
                return None
        candidates = [card_id for card_id, definition in cards.items() if normalized(definition.name) in titles
                      and (possible_forms is None or card_id in possible_forms)]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates and titles:
            folded = {fold_card_ocr_title(title) for title in titles}
            candidates = [cid for cid, card in cards.items()
                          if fold_card_ocr_title(normalized(card.name)) in folded
                          and (possible_forms is None or cid in possible_forms)]
            if len(candidates) == 1:
                return candidates[0]
        if matched:
            return read_form(matched)
        return (read_form(prediction.card_id) if possible_forms is None and prediction.accepted
                 and prediction.card_id in cards else None)

    def read_overlapping_hand(self, image, fresh, branches):
        """Resolve a clipped title from the forms that can be present after the action."""
        frame = fresh.observation
        forms = Counter()
        for branch in branches:
            state = branch.state
            if (not state.terminal and state.pending_choice is None
                    and (state.turns_remaining, state.stamina, state.genki) == (frame.turns, frame.stamina, frame.genki)
                    and len(state.deck.hand) == len(frame.hand)):
                forms |= Counter(card.card_id for card in state.deck.hand)
        return self.read_overlapping_forms(image, fresh, forms)

    def read_overlapping_forms(self, image, fresh, forms):
        if not forms:
            return fresh
        frame = fresh.observation
        hand = list(frame.hand)
        known = set(forms)
        titles = self.live_hand_titles(image, fresh.boxes)
        if isinstance(forms, Counter) and (excess := Counter(hand) - forms):
            unused = forms - Counter(hand)
            repairs = {}
            for slot, card in enumerate(hand):
                if card not in excess or card not in known:
                    continue
                replacement = self.hiro_card_id(image, fresh.boxes[slot], slot, possible_forms=unused)
                if replacement is not None and self.prefer_live_hand_title(replacement, titles.get(slot)) == replacement:
                    repairs[slot] = replacement
            # Each repaired occurrence needs its own visual evidence and space in the observed zones.
            if Counter(hand[slot] for slot in repairs) <= excess and Counter(repairs.values()) <= unused:
                for slot, card in repairs.items():
                    hand[slot] = card
        for slot, box in enumerate(fresh.boxes):
            right = fresh.boxes[slot + 1][0] if slot + 1 < len(fresh.boxes) else None
            clipped = right is not None and right < box[0] + box[2] - 4
            if isinstance(forms, Counter):
                other = Counter(hand)
                other[hand[slot]] -= 1
                admissible = set(forms - other)
            else:
                admissible = known
            if hand[slot] not in known:
                read = self.card_id
            elif clipped and hand[slot] in admissible:
                read = self._gkh_card_id
            else:
                continue
            card = read(image, box, slot, visible_right=right if clipped else None,
                        possible_forms=admissible, dimmed=slot not in frame.playable)
            if card is not None:
                hand[slot] = self.prefer_live_hand_title(card, titles.get(slot))
        return replace(fresh, observation=replace(frame, hand=tuple(hand)))

    def hand_boxes(self, image):
        result = self.recognition('ProduceRecognitionCards', image)
        rows = result.all_results if result else ()
        expanded = [SimpleNamespace(box=row.box, score=row.score, label='cards') for row in rows
                    if row.label in ('cards', 'suggestions', 'useless')]
        boxes = tuple(tuple(box) for box, _ in self._hif_card_boxes(expanded))
        playable = tuple(i for i, box in enumerate(boxes) if any(
            row.label in ('cards', 'suggestions') and self._hif_duplicate_box(box, row.box) for row in rows))
        anchored = anchor_exam_hand(image, rows, self)
        if anchored is not None:
            return tuple(row.box for row in anchored), tuple(i for i, row in enumerate(anchored)
                                                             if row.label != 'useless')
        return boxes, playable

    def read_progress_signature(self, image, *, screen=None):
        """Read Hiro's round, HUD numbers and hand positions without identifying the cards."""
        if screen is not None:
            live = screen.observation
            hud, boxes = (live.turns, live.stamina, live.genki), screen.boxes
        else:
            turns = self.remaining_turns(image)
            if turns is None:
                return None
            boxes, _ = self.hand_boxes(image)
            if not boxes and not self.recognition('ProduceRecognitionNoCards', image):
                return None
            hud = turns, self.read_stamina(image), self.read_genki(image)
        return hud, tuple((_center(box)[0] // 8, (box[1] + box[3]) // 8) for box in boxes)

    def is_trouble_card(self, card_id):
        return getattr(self.kernel.cards.get(card_id), 'card_type', None) == 'trouble'

    @staticmethod
    def _hif_has_select(context, image, box):
        if BattleRecognitionMixin._hif_has_select(context, image, box):
            return True
        markers, _ = read_exam_select_markers(context, image)
        return any(abs(_center(marker)[0] - _center(box)[0]) <= 70 for marker in markers)

    def remaining_turns(self, image):
        turns = read_entry_turn(self.context, image)
        if turns is None:
            return None
        for roi in ((80, 112, 87, 52), (98, 120, 53, 34)):
            bonus = {int(match[1]) for row in self.ocr(image, roi=roi, expected=r'^\+[1-9]\d?$')
                     if row.score >= .95 and (match := re.fullmatch(r'\+([1-9]\d?)', row.text))}
            if len(bonus) == 1:
                return turns + next(iter(bonus))
        return turns

    def read_status_value(self, image, roi, *, percent, turns=False):
        expected = r'^\s*\d{1,3}\s*[%％]\s*$' if percent else r'^\s*\d{1,3}\s*$'
        status = hiro_module('vision.status') if turns else None
        if status is not None:
            expected = [expected, r'^\s*[0-9０-９ろOolI|S]{1,2}\s*ターン\s*$']
        values = set()
        for row in self.ocr(image, roi=roi, expected=expected):
            if row.score < .95:
                continue
            text = compact_ocr_text(row.text)
            if status is not None and (duration := status._turns_from_text(text)) is not None:
                values.add(duration)
            else:
                match = re.fullmatch(r'(\d{1,3})%' if percent else r'(\d{1,3})', text)
                if match is not None:
                    values.add(int(match[1]))
        return next(iter(values)) if len(values) == 1 else None

    def read_live_drinks(self, image, blocked_slot=None):
        observed = self.drink_reader.read_slots(image)
        occupied = {}
        for slot, row in enumerate(observed):
            if row['state'] == 'OCCUPIED':
                occupied[slot] = row
            elif row['state'] == 'UNKNOWN':
                if self.live_screen is None or slot == blocked_slot:
                    return None
                previous = self.live_screen.drinks.get(slot)
                if previous is not None:
                    # The current action did not touch this bottle. Keep its identity
                    # in the observation but require a readable icon before using it.
                    occupied[slot] = dict(previous, available=False)
        return occupied

    def read_audition_attribute(self, image, roi):
        labels = self.ocr(image, roi=roi, expected=r'ボーカル|ダンス|ビジュアル')
        names = dict(zip(('ボーカル', 'ダンス', 'ビジュアル'), ATTRIBUTES, strict=True))
        found = {attr for row in labels for name, attr in names.items() if name in compact_ocr_text(row.text)}
        return next(iter(found)) if len(found) == 1 else None

    def read_exam_position(self, image):
        rank = read_idol_road_live_ranking(image, self.ranking_number)
        attribute = self.read_audition_attribute(image, (95, 48, 141, 49))
        return attribute, rank['score'] if rank is not None else None

    def observe_live(self, image, *, blocked_drink_slot=None):
        if image is not getattr(self, 'image', None):
            return self._read_live(image, blocked_drink_slot)
        basis = self.live_screen
        cached = self._live_reads.get(blocked_drink_slot)
        if cached is not None and cached[0] is basis:
            return cached[1]
        observed = self._read_live(image, blocked_drink_slot)
        self._live_reads[blocked_drink_slot] = basis, observed
        return observed

    def _read_live(self, image, blocked_drink_slot):
        skip = self.target('ProduceRecognitionSkipRound', image, HIF_SKIP_RECOGNITION)
        if skip is None or not self.recognition('ProduceRecognitionHealthFlag', image):
            return None
        boxes, playable = self.hand_boxes(image)
        if not hand_row_ready(boxes):
            return None
        if not boxes and not self.recognition('ProduceRecognitionNoCards', image):
            return None
        attribute, score = self.read_exam_position(image)
        turns = self.remaining_turns(image)
        stamina = self.read_stamina(image)
        genki = self.read_genki(image)
        if None in (turns, stamina, genki, attribute, score) or score < 0:
            return None
        hand = tuple(self.card_id(image, box, i, dimmed=i not in playable, visible_right=boxes[i + 1][0]
                     if i + 1 < len(boxes) and boxes[i + 1][0] < box[0] + box[2] - 4 else None)
                     for i, box in enumerate(boxes))
        titles = self.live_hand_titles(image, boxes)
        hand = tuple(self.prefer_live_hand_title(cid, titles.get(i)) for i, cid in enumerate(hand))
        if any(cid is None for cid in hand):
            return None
        # Hiro treats Trouble cards as playable despite their naturally gray art.
        playable = tuple(i for i, cid in enumerate(hand) if i in playable or self.is_trouble_card(cid))
        visible_bottles = self.read_live_drinks(image, blocked_drink_slot)
        if visible_bottles is None:
            return None
        drinks = tuple((i, row['drink_id']) for i, row in visible_bottles.items())
        available = tuple(i for i, row in visible_bottles.items() if row['available'] is True)
        x, y, width, _ = skip
        recovery = self.numbers(image, (x + 13, y + 9, width - 26, 33))
        if recovery is None and isinstance(self.kernel, TowerCardKernel):
            recovery = tower_skip_recovery()
        statuses = read_tower_status_values(image, self.read_status_value)
        status_slots = ()
        if getattr(self, 'hiro_policy_route', False):
            cache = getattr(self, '_hiro_status_cache', None)
            if cache is None:
                cache = self._hiro_status_cache = hiro_module('vision.region_cache').RegionCache()
            status_slots = read_hiro_statuses(image, self.ocr, cache=cache)
            statuses.update(hiro_primary_statuses(status_slots))
        observation = TowerFrame(self.frame_id, turns, attribute, score, stamina, genki,
            hand, drinks, playable, available, skip_recovery=recovery, statuses=tuple(statuses.items()),
            status_slots=status_slots)
        return TowerScreen(observation, boxes, skip, visible_bottles)

    def play_card(self, slot, *, before=None, action=None):
        dispatch_image = None

        def current_target(image):
            nonlocal dispatch_image
            if (ready := self.current_card_target(image, self.live_screen, slot)) is not None:
                dispatch_image = image
                return ready
            return self.reobserve_pending(image, before, action)

        ready = self.poll(current_target, 'current-card-target')
        if isinstance(ready, ExamReobserved):
            return ready
        first, slot, target = ready

        def selected(image):
            boxes, _ = self.hand_boxes(image)
            current = [box for box in boxes if abs(_center(box)[0] - _center(target)[0]) <= self.HIF_PREVIEW_X_TOLERANCE
                       and self._hif_has_select(self.context, image, box)]
            return (current[0] if len(current) == 1 else None), bool(current)

        def stopped():
            return self.context.tasker.stopping

        def retry_target(image):
            nonlocal target
            anchor = first
            if dispatch_image is not None:
                # Hiro compares against the frame used for the first click.
                anchor = replace(first, observation=replace(first.observation,
                    stamina=self.read_stamina(dispatch_image), genki=self.read_genki(dispatch_image)))
            retried = self.retry_selection_target(image, anchor, slot)
            if retried is not None:
                target = retried
            return retried

        def observe_selection(image):
            return selected(image)[0] or self.reobserve_pending(image, before, action, first_tap=True)

        def refresh():
            if not stopped():
                image = self.capture()
                return observe_selection(image) or self.reobserve_no_effect(before, action, self.observe_live(image))
            return None

        def reconcile():
            if self.native_session is None or before is None or stopped():
                raise ExamNoProgress('idol_road_no_progress:target-select')
            return ExamFeedback(observe_selection, refresh).wait(self.poll, 'target-select')

        def tap(box, *, purpose):
            self.click(_center(box), purpose=purpose)
            if purpose == 'confirm-card':
                self.card_confirmation = ExamCardConfirmation(tuple(box), time.monotonic())

        return select_exam_card(target, tap=tap,
                                capture=self.capture, stopping=stopped, read_select=selected,
                                retry_target=retry_target, reconcile=reconcile)

    def recheck_confirmed_card(self, image, confirmation):
        """Hiro's SELECT grace and spatial recheck for a pending confirmation."""
        def marker(box):
            return _center(box)[0], box[1] + box[3]

        def selected(frame, anchor):
            if _battle_dialog_header(frame) is not None:
                return None
            boxes, _ = self.hand_boxes(frame)
            x, y = marker(anchor)
            matched = []
            for box in boxes:
                bx, by = marker(box)
                if abs(bx - x) <= 30 and abs(by - y) <= 30 and self._hif_has_select(self.context, frame, box):
                    matched.append(box)
            if len(matched) != 1 or self.remaining_turns(frame) != before.turns:
                return None
            resources = self.read_stamina(frame), self.read_genki(frame)
            if same_readable_resources((before.stamina, before.genki), resources):
                return matched[0]
            return None

        if time.monotonic() - confirmation.sent_at < 1.5 or self.context.tasker.stopping:
            return None
        before = self.live_screen.observation
        return recheck_persistent_select(image, confirmation.box, selected=selected, capture=self.capture,
                                         stopping=lambda: self.context.tasker.stopping)

    def reobserve_pending(self, image, before, action, *, first_tap=False):
        if self.native_session is None or before is None or before.decision_kind is not DecisionKind.LIVE:
            return None
        fresh = self.observe_live(image)
        if fresh is None:
            return None
        previous = self.live_screen.observation
        frame = fresh.observation
        if self.same_live(fresh, self.live_screen) and (frame.playable, frame.available_drinks) == (
                previous.playable, previous.available_drinks):
            return None
        step = None
        if first_tap:
            # A first tap can itself play a card. Only report it if the resulting
            # position is supported independently of an automatic turn advance.
            step = self.native_session.observe_frame(before, action, frame)
            candidate = self.native_session.candidate
            played_delta = self.native_session.observed_delta if step is not None else {}
            resources = {'stamina', 'genki'}
            played_values_match = step is not None and resources.isdisjoint(played_delta)
            passive = (frame.turns < previous.turns and (frame.stamina, frame.genki) ==
                       (previous.stamina, previous.genki))
            request = self.native_session.observed_position(before, frame) if passive else None
            passive_values_match = request is not None and resources.isdisjoint(self.native_session.observed_delta)
            if played_values_match == passive_values_match:
                return None
            if passive_values_match:
                step = None
            else:
                request = None
                self.native_session.candidate = candidate
                self.native_session.observed_delta = played_delta
        else:
            observed = self.reobserve_visible(image, before, fresh)
            request, fresh = observed if observed is not None else (None, fresh)
            frame = fresh.observation
        if request is None and step is None:
            return None
        next_request = request if request is not None else step.request
        self.accept_live_screen(fresh, next_request)
        self.log(dict(event='idol_road_pending_action_reobserved', frame_id=frame.frame_id,
                      action=action.kind.value, first_tap_sent=first_tap, action_applied=step is not None,
                      turns=frame.turns, hand=frame.hand, score=frame.score,
                      observed_minus_model=self.native_session.observed_delta))
        return ExamReobserved(request) if request is not None else step

    def accept_live_screen(self, fresh, request):
        if request.state.turn > len(self.observed_types):
            self.observed_types.append(ATTRIBUTES.index(fresh.observation.attribute) + 1)
        self.live_screen = fresh

    def reobserve_visible(self, image, request, fresh):
        if self.native_session is None:
            return None
        frame = fresh.observation
        observed = self.native_session.observed_position(request, frame)
        if observed is None:
            forms = self.native_session.idle_hand_forms(request, frame)
            corrected = self.read_overlapping_forms(image, fresh, forms)
            if corrected != fresh:
                fresh = corrected
                observed = self.native_session.observed_position(request, fresh.observation)
        return (observed, fresh) if observed is not None else None

    def reobserve_no_effect(self, before, action, fresh, *, unconfirmed_image=None):
        if fresh is None or self.native_session is None:
            return None
        frame, prior = fresh.observation, self.live_screen.observation
        if (frame.attribute, frame.drinks) != (prior.attribute, prior.drinks):
            return None
        if unconfirmed_image is not None:
            # Use the ordered slots and the card identities Maa binds to them
            # as the hand signature in Hiro's missed-selection comparison.
            if frame.hand != prior.hand and len(frame.hand) == len(prior.hand):
                fresh = self.read_overlapping_forms(unconfirmed_image, fresh, Counter(prior.hand))
                frame = fresh.observation
            old = (prior.turns, prior.stamina, prior.genki), prior.hand
            new = (frame.turns, frame.stamina, frame.genki), frame.hand
            same = hiro_module('exec.exam')._lite_same(old, new)
        else:
            same = (frame.turns, frame.stamina, frame.genki, frame.hand, frame.score) == (
                prior.turns, prior.stamina, prior.genki, prior.hand, prior.score)
        if not same:
            return None
        request = self.native_session.observed_position(before, frame)
        if request is None:
            return None
        self.accept_live_screen(fresh, request)
        kind = 'skip' if action.kind is ActionKind.END_TURN else 'card_select'
        self.log(dict(event=f'idol_road_{kind}_no_effect', frame_id=frame.frame_id,
                      turns=frame.turns, hand=frame.hand, score=frame.score))
        return ExamReobserved(request, dispatch_no_effect=True)

    def current_card_target(self, image, first, slot):
        """Bind the intended card to a current control before the first click."""
        before = first.observation
        if self.current_skip(image, before.turns) is None:
            return None
        boxes, playable = self.hand_boxes(image)
        if len(boxes) != len(first.boxes) or not hand_row_ready(boxes):
            return None

        target_id = before.hand[slot]
        titles = self.live_hand_titles(image, boxes)
        # Try the expected physical slot first. Identity comparison catches a
        # same-width reorder, which geometry alone cannot see.
        def read_id(index):
            box = boxes[index]
            right = boxes[index + 1][0] if index + 1 < len(boxes) else None
            if right is not None and right < box[0] + box[2] - 4:
                card_id = self.card_id(image, box, index, visible_right=right, possible_forms=set(before.hand))
            else:
                card_id = self.card_id(image, box, index)
                if card_id not in before.hand:
                    card_id = self.card_id(image, box, index, possible_forms=set(before.hand))
            return self.prefer_live_hand_title(card_id, titles.get(index))

        at_slot = read_id(slot)
        if at_slot == target_id and (slot in playable or self.is_trouble_card(target_id)):
            return replace(first, boxes=boxes), slot, boxes[slot]
        if before.hand.count(target_id) != 1:
            return None
        observed = tuple(at_slot if index == slot else read_id(index) for index in range(len(boxes)))
        expected = Counter(before.hand)
        if Counter(observed) != expected:
            current = TowerScreen(replace(before, hand=observed, playable=playable), boxes, first.skip, first.drinks)
            observed = self.read_overlapping_forms(image, current, expected).observation.hand
            if Counter(observed) != expected:
                return None
        actual = observed.index(target_id)
        if actual not in playable and not self.is_trouble_card(target_id):
            return None
        self.log(dict(event='idol_road_current_card_rebound', previous_slot=slot, current_slot=actual,
                      card_id=target_id, frame_id=self.frame_id))
        return replace(first, boxes=boxes, observation=replace(before, hand=observed)), actual, boxes[actual]

    def current_skip(self, image, turns):
        skip = self.target('ProduceRecognitionSkipRound', image, HIF_SKIP_RECOGNITION)
        return skip if skip is not None and self.remaining_turns(image) == turns else None

    def advance_exam_animation(self, image, *, ranking=False):
        purpose = 'advance-tower-finish-animation' if ranking else 'advance-tower-animation'
        return self.exam_recovery.advance_animation(self.context, image,
            is_animating=lambda: ranking or (not self.target('ProduceRecognitionSkipRound', image, HIF_SKIP_RECOGNITION)
                and self.recognition('ProduceRecognitionHealthFlag', image)),
            click=lambda point: self.click(point, purpose=purpose))

    def current_drink_target(self, image, action, turns):
        if self.current_skip(image, turns) is None:
            return None
        drink = self.drink_reader.read_slots(image)[action.source_slot]
        if drink['state'] == 'OCCUPIED' and drink['available'] is True and drink['drink_id'] == action.drink_id:
            return drink['center']
        return None

    def retry_selection_target(self, image, first, slot):
        boxes = self.retry_selection_boxes(image, first)
        if self.native_session and (fresh := self.observe_live(image)) is not None:
            before = first.observation
            fresh = self.read_overlapping_forms(image, fresh, Counter(before.hand))
            after = fresh.observation
            if (not same_readable_resources((before.stamina, before.genki), (after.stamina, after.genki))
                    or (before.turns, before.attribute, before.drinks, Counter(before.hand)) != (
                        after.turns, after.attribute, after.drinks, Counter(after.hand))):
                return None
            if before.hand != after.hand:
                card = before.hand[slot]
                if after.hand.count(card) != 1:
                    return None
                slot = after.hand.index(card)
            boxes = fresh.boxes
        if not boxes or any(self._hif_has_select(self.context, image, box) for box in boxes):
            return None
        return boxes[slot]

    def retry_selection_boxes(self, image, first):
        """Use Hiro's lightweight progress check on an unconfirmed first click."""
        before = first.observation
        if self.remaining_turns(image) != before.turns:
            return None
        boxes, _ = self.hand_boxes(image)
        if (not boxes or len(boxes) != len(first.boxes)
                or any(abs(old - new) > 8 for left, right in zip(boxes, first.boxes, strict=True)
                       for old, new in zip(_center(left), _center(right), strict=True))):
            return None
        stamina = self.read_stamina(image)
        genki = self.read_genki(image)
        if not same_readable_resources((before.stamina, before.genki), (stamina, genki)):
            return None
        return boxes

    def inspect_formation_effects(self, image, layer):
        icons = cv2.cvtColor(image[45:106, 373:520], cv2.COLOR_BGR2HSV)
        if not (icons[:, :, 1] > 130).any():
            return {}
        self.click((409, 75), purpose='open-formation-effects')

        def read(frame):
            page = read_formation_effects_page(frame, self.ocr(frame))
            return page if page is not None and page.layer == layer else None

        page = self.poll(read, 'formation-effects')
        rows = {row.turn: row for row in page.rows}
        variants = tower_stage_descriptions(layer)
        try:
            completed = complete_formation_effects(rows, variants)
            while completed is rows and not page.at_bottom:
                job = self.context.tasker.controller.post_swipe(550, 846, 550, 410, 400).wait()
                self.log(dict(event='idol_road_maa_input', purpose='scroll-formation-effects',
                              frame_id=self.frame_id, job_id=job.job_id, succeeded=job.succeeded))
                if not job.succeeded:
                    raise ValueError('idol_road_formation_effects_scroll_failed')

                def advanced(frame):
                    observed = read(frame)
                    return observed if observed is not None and observed != page else None

                page = self.poll(advanced, 'formation-effects-scroll')
                rows.update({row.turn: row for row in page.rows})
                completed = complete_formation_effects(rows, variants)
        finally:
            self.click(page.close, purpose='close-formation-effects')
        self.poll(lambda frame: frame if (overview := read_formation_overview(self.formation_ocr(frame)))
                  and overview.page == 1 and overview.layer == layer else None, 'formation-after-effects')
        if not completed:
            raise ValueError('idol_road_native_gimmick_variant_unread')
        self.log(dict(event='idol_road_formation_effects', layer=layer, rows=[vars(row) for row in completed.values()],
                      master_completed=[turn for turn, row in completed.items() if rows.get(turn) != row],
                      unmapped=[row.turn for row in completed.values() if not tower_field_rule(row.description)]))
        return completed

    @staticmethod
    def same_live(left, right):
        a, b = left.observation, right.observation
        return (a.turns, a.attribute, a.hand, a.drinks, a.score, a.stamina, a.genki, a.statuses, a.status_slots) == (
                b.turns, b.attribute, b.hand, b.drinks, b.score, b.stamina, b.genki, b.statuses, b.status_slots)

    def observed_item_uses(self):
        limited = frozenset(item for item, limit in self.item_uses.items() if limit is not None)
        if not limited:
            return {}
        try:
            return read_bound_item_uses(self.context, self.image, _data('hif_item_icons'), limited)
        except (OSError, ValueError, cv2.error):
            return {}

    def opening_item_uses(self):
        observed = self.observed_item_uses()
        if not self.native_rules:
            observed = {item: count for item, count in observed.items() if count <= self.item_uses[item]}
        return self.item_uses | observed

    def submit(self, before, action):
        self.card_confirmation = None
        if action.kind is ActionKind.PLAY_CARD:
            return self.play_card(action.source_slot, before=before, action=action)
        elif action.kind is ActionKind.USE_DRINK:
            target = self.poll(lambda image: self.current_drink_target(image, action, self.live_screen.observation.turns)
                               or self.reobserve_pending(image, before, action),
                               'current-drink-target')
            if isinstance(target, ExamReobserved):
                return target
            source = self.live_screen
            self.click(target, purpose='open-drink')

            def retry_open(image):
                if self.context.tasker.stopping:
                    return
                current = self.observe_live(image)
                if current is not None and self.same_live(current, source):
                    point = self.current_drink_target(image, action, source.observation.turns)
                    if point is not None:
                        self.click(point, purpose='retry-open-drink')

            probe = exam_drink_open_probe(lambda image: upstream_drink_actions(self.context, image),
                lambda actions: actions.get('use'), retry_open,
                dialog_visible=lambda image: _battle_dialog_header(image) is not None)
            try:
                use = self.poll(probe, 'drink-detail', duration=8.)
            except ExamNoProgress:
                if self.context.tasker.stopping:
                    raise
                frame = self.capture()
                if self.native_session and (observed := self.reobserve_no_effect(before, action, self.observe_live(frame))):
                    self.unplayable_drinks.add((before.state.turn, action.source_slot))
                    return observed
                raise
            self.click(_center(use), purpose='use-drink')
        elif action.kind is ActionKind.END_TURN:
            skip = self.poll(lambda image: self.current_skip(image, self.live_screen.observation.turns)
                             or self.reobserve_pending(image, before, action), 'current-skip')
            if isinstance(skip, ExamReobserved):
                return skip
            self.click(_center(skip), purpose='skip-turn')
        elif before.decision_kind is DecisionKind.SECONDARY:
            self.submit_choice(before, action)
        else:
            raise ValueError('idol_road_action_unsupported:' + action.kind.value)

    def submit_choice(self, before, action):
        kind = before.state.pending_choice.action_kind
        if action.kind is ActionKind.FINISH_SELECTION:
            def cancel(image):
                buttons = battle_choice_buttons(image)
                if buttons is not None:
                    return buttons[0]
                if self.choice_visible(image, kind):
                    return self.choice_text(image, 'IdolRoadChoiceFinish',
                        r'キャンセル|スキップ|選択しない', (100, 1100, 520, 150))
                return None

            finish = self.poll(cancel, 'choice-finish')
            self.click(_center(finish), purpose='finish-choice')
            return
        slot = dict(before.card_slots)[action.card_instance_id]
        self.click(_center(self.choice_boxes[slot]), purpose='select-choice-card')

        def confirm(image):
            picked = (self._hif_has_select(self.context, image, self.choice_boxes[slot])
                      if self.choice_is_hand else _choice_selected_slot(image, self.choice_boxes) == slot)
            if not picked:
                return None
            if kind is ActionKind.MOVE_CARD:
                native = self.target('ProduceMoveCards', image)
                if native is not None:
                    return native
            buttons = battle_choice_buttons(image)
            if buttons is not None:
                return buttons[1]
            if not self.choice_visible(image, kind):
                return None
            return self.choice_text(image, 'IdolRoadChoiceCommit', _CHOICE_CONFIRMS[kind],
                                    (200, 1105, 350, 150))

        target = self.poll(confirm, 'choice-confirm')
        self.click(_center(target), purpose='confirm-choice')

    def choice_text(self, image, node, pattern, roi):
        result = self.recognition(node, image,
            dict(recognition='OCR', roi=list(roi), expected=f'^(?:{pattern})$'))
        rows = tuple(result.filtered_results) if result else ()
        return tuple(rows[0].box) if len(rows) == 1 else None

    def choice_visible(self, image, kind):
        if kind is ActionKind.MOVE_CARD and (self.recognition('ProduceRecognitionChooseMoveCards', image)
                                             or self.target('ProduceMoveCards', image)):
            return True
        header = _CHOICE_HEADERS.get(kind)
        return header is not None and self.choice_text(image, 'IdolRoadChoiceHeading', header, (0, 15, 720, 130)) is not None

    def native_target_choice_visible(self, image, heading):
        # The game also uses the shared selection dialog for native target
        # effects. The pending effect supplies the operation on those pages.
        if battle_choice_buttons(image) is not None:
            return True
        top = (0, 15, 720, 130)
        if self.choice_text(image, 'IdolRoadChoiceHeading', heading, top) is not None:
            return True
        if self.choice_text(image, 'IdolRoadChoiceHeading', r'選択.{0,1}', (35, 28, 175, 57)) is None:
            return False
        return (self.choice_text(image, 'IdolRoadChoiceCancel', r'.{0,2}キャンセル', (60, 1105, 300, 150)) is not None
                and self.choice_text(image, 'IdolRoadChoiceCommit', r'.{0,2}決定', (365, 1105, 285, 150)) is not None)

    def pending_dialog(self, image, action, *, slow=False):
        """Complete the confirmation for the action already sent to the game."""
        if action.kind not in _BATTLE_CONFIRMATIONS:
            return None
        band = _battle_dialog_header(image)
        if band is None and not slow:
            return None
        if band is None:
            rows = self.ocr(image)
        else:
            top, bottom = band
            rows = self.ocr(image, roi=(25, top, 670, bottom - top))
        title = ''.join(re.sub(r'\s+', '', row.text) for row in rows)
        if any(word in title for word in ('リタイア', '破棄')):
            return self.choice_text(image, 'IdolRoadBattleCancel', r'.{0,2}キャンセル', (20, 600, 680, 664))
        heading, button = _BATTLE_CONFIRMATIONS[action.kind]
        if not re.search(heading, title):
            return None
        return self.choice_text(image, 'IdolRoadBattleConfirm', rf'.{{0,2}}(?:{button})', (20, 1000, 680, 265))

    def read_choice(self, image, before, branches, drinks, *, panel=False):
        kinds = {branch.state.pending_choice.action_kind for branch in branches if branch.state.pending_choice is not None}
        shown_kinds = list(kinds) if panel and len(kinds) == 1 else [kind for kind in kinds if self.choice_visible(image, kind)]
        if len(shown_kinds) != 1:
            return None
        kind, = shown_kinds
        candidates = tuple(branch for branch in branches if branch.state.pending_choice is not None
                           and branch.state.pending_choice.action_kind is kind)
        counts = {len(branch.state.pending_choice.candidate_instance_ids) for branch in candidates}
        expected = next(iter(counts)) if len(counts) == 1 else None
        as_hand = False
        try:
            boxes = hif_choice_grid(image, expected_count=expected)
        except ValueError:
            if kind is not ActionKind.HOLD_CARD:
                return None
            boxes, _ = self.hand_boxes(image)
            if not boxes or len(boxes) != expected:
                return None
            as_hand = True
        if before.decision_kind is DecisionKind.SECONDARY:
            selected = (any(self._hif_has_select(self.context, image, box) for box in boxes)
                        if as_hand else _choice_selected_slot(image, boxes) is not None)
            if selected:
                return None
        shown = tuple(self.card_id(image, box, i) for i, box in enumerate(boxes))
        if any(card is None for card in shown):
            return None
        step = tower_choice_step(before, candidates, self.frame_id, shown, drink_slots=drinks)
        if step:
            self.choice_boxes, self.choice_is_hand = boxes, as_hand
        return step

    def read_result_page(self, image):
        overview = getattr(self.loadout, 'overview', None)
        return read_idol_road_result_frame(self.ocr, image, idol_name=getattr(overview, 'idol_name', None),
                                           hiro=getattr(self, 'hiro_policy_route', False))

    def read_unscored_failure(self, image):
        from .exam_hud import exam_failed_header

        if exam_failed_header(image, self.ocr):
            return dict(rank=None, score=None, failed=True)
        return None

    def result_page_ready(self, image):
        return bool(self.recognition('IdolRoadRetry', image) or self.recognition('IdolRoadAutoNext', image))

    def read_hiro_tower_settlement(self, image):
        from .exam_hud import exam_failed_header
        from .exam_loop import ExamSettlement

        if self.recognition('ProduceRecognitionHealthFlag', image):
            return None
        if self.result_page_ready(image):
            result = self.read_result_page(image)
            if result is not None:
                return ExamSettlement(result['rank'], result['score'], None, failed=result['rank'] > 3)
            if exam_failed_header(image, self.ocr):
                return ExamSettlement(None, None, None, failed=True)
            if self.target('IdolRoadAutoNext', image):
                boxes = [(''.join(row.text.split()), (x, y, x + w, y + h))
                         for row in self.ocr(image) for x, y, w, h in (row.box,)]
                page = hiro_module('vision.tower').tower_page(image, boxes)
                if page is not None and page[0] == 'TOWER_RANKING' and not page[1].get('failed'):
                    return ExamSettlement(page[1].get('my_rank'), None, None)
        if self.target('IdolRoadBackToList', image):
            self.screen_clear = read_tower_clear_result(
                image, self.ocr(image), ocr=self.ocr, layer=getattr(self, 'last_settled_layer', None))
            if self.screen_clear is not None:
                return ExamSettlement(None, None, None)
        return None

    def run_hiro_screen_tower(self, *, retry):
        """Resume the visible exam with Hiro when no usable model start exists."""
        from .hiro_maa_exam import run_maa_screen_exam

        self.screen_clear = None
        return run_maa_screen_exam(self, self.read_hiro_tower_settlement, label='tower-x', retry=retry, capture=self.capture)

    def confirm_ranking_update(self, *, layer=None, first=None):
        """Dismiss a stable ranking popup and read back the same settlement page."""
        stable, sent_points, previous_result = None, None, None
        sent_at, repeated = None, False

        def observe_popup(image, box):
            nonlocal stable, sent_points, sent_at, repeated
            if image.shape != (1280, 720, 3):
                stable = None
                return
            points = image[630:693, 235:485].copy()
            now = time.monotonic()
            if stable is None or box != stable[0] or not _same_ranking_points(points, stable[1]):
                stable = box, points, now
                return
            if now - stable[2] < _RANKING_UPDATE_STABLE_S:
                return
            if sent_at is not None:
                if repeated or now - sent_at < _RANKING_UPDATE_RETRY_S or not _same_ranking_points(points, sent_points):
                    return
                repeated = True
            self.click(_center(box), purpose='tower-ranking-update-ok-repeat' if repeated else 'tower-ranking-update-ok')
            sent_points, sent_at = points, time.monotonic()
            stable = None

        def read(image):
            nonlocal stable, previous_result
            update = self.recognition('IdolRoadRankingUpdate', image)
            if update is not None:
                previous_result = None
                observe_popup(image, tuple(update.box))
                return None
            stable = None
            result = None
            if self.target('IdolRoadBackToList', image):
                result = read_tower_clear_result(image, self.ocr(image), ocr=self.ocr, layer=layer)
            if result is not None and result == previous_result:
                self.log(dict(event='idol_road_ranking_update_confirmed', layer=result[0],
                              clicks=2 if repeated else int(sent_at is not None)))
                return result, image
            previous_result = result
            return None

        if first is not None:
            observe_popup(*first)
        return self.poll(read, 'tower-ranking-update-confirm')

    def settle_hiro_tower(self, result, attempts, *, screen_run=None):
        """Resolve the Maa page corresponding to Hiro's result verdict."""
        layer = self.loadout.overview.layer if screen_run is None else getattr(self, 'last_settled_layer', None)
        failed = getattr(result, 'failed', False) or (result.rank is not None and result.rank > 3)
        cleared = not failed and (result.rank is not None or screen_run is not None)
        stars, overview = 0, False
        update_confirmed = False

        def ranking_update(image):
            nonlocal update_confirmed
            if not cleared:
                return None
            update = self.recognition('IdolRoadRankingUpdate', image)
            if update is None:
                return None
            if update_confirmed:
                raise ExamNoProgress('idol_road_no_progress:tower-ranking-update-confirm')
            confirmed = self.confirm_ranking_update(layer=layer, first=(image, tuple(update.box)))
            update_confirmed = True
            return confirmed

        def result_button(node, image):
            confirmed = ranking_update(image)
            return self.target(node, confirmed[1] if confirmed is not None else image)

        ready = self.screen_clear if screen_run is not None else None
        if cleared and ready is not None:
            layer, stars = ready
        elif cleared:
            target = self.poll(lambda image: self.target('IdolRoadAutoNext', image), 'tower-ranking-next')
            self.click(_center(target), purpose='tower-ranking-next')

            previous_clear = None
            next_sent = time.monotonic()
            next_repeated = False

            def clear_page(image):
                nonlocal previous_clear, next_repeated
                confirmed = ranking_update(image)
                if confirmed is not None:
                    return confirmed[0], False
                rows = self.ocr(image)
                settled = read_tower_clear_result(image, rows, ocr=self.ocr, layer=layer)
                if settled is not None and settled[1] == 3:
                    return settled, False
                if settled is not None and settled == previous_clear:
                    return settled, False
                previous_clear = settled
                if settled is not None:
                    return None
                floor = read_floor_overview(image, rows, self.target('IdolRoadStartFlag', image))
                if floor is not None and floor.number == layer and floor.stars == 3:
                    return (layer, 3), True
                if not next_repeated and time.monotonic() - next_sent >= hiro_module('runtime.tower_play').RESULT_REDO_S:
                    again = self.target('IdolRoadAutoNext', image)
                    current = self.read_result_page(image) if again else None
                    if current is not None and (current['rank'], current['score']) == (result.rank, result.score):
                        self.click(_center(again), purpose='tower-ranking-next-repeat')
                        next_repeated = True
                return None

            (layer, stars), overview = self.poll(clear_page, 'tower-clear-result')
        self.last_settled_layer = layer
        screen_idol = getattr(self, 'screen_tower_idol', None)
        cache = getattr(self, '_tower_result_cache', None)
        receipt = cache.active_id() if cache is not None else None
        if screen_run is None:
            verdict = attempts.hiro_result(self.loadout, result, stars=stars, receipt=receipt)
        elif layer is not None and screen_idol is not None:
            verdict = attempts.stage_result(attempts.stage_path(screen_idol, layer), layer, result,
                                            stars=stars, receipt=receipt)
        else:
            if layer is not None and None in screen_run.tries:
                screen_run.tries[layer] = screen_run.tries.pop(None)
                screen_run.best[layer] = screen_run.best.pop(None)
            verdict = attempts.verdict(screen_run, layer, cleared, stars) if attempts else screen_run.after_result(
                layer, cleared=cleared, stars=stars)
        self.log(dict(event='hiro_tower_result', layer=layer, rank=result.rank, score=result.score,
                      stars=stars, verdict=verdict, failed=failed))
        if verdict == 'retry':
            target = self.poll(lambda image: result_button('IdolRoadRetry', image), 'tower-result-retry')
            self.click(_center(target), purpose='tower-result-retry')
            self._tower_retry_result = (time.monotonic(), layer, failed, stars, result)
        elif verdict == 'next' and not overview:
            target = self.poll(lambda image: result_button('IdolRoadBackToList', image), 'tower-result-overview')
            self.click(_center(target), purpose='tower-result-overview')
            start = time.monotonic()
            repeated = False

            def departure(image):
                nonlocal repeated
                if ranking_update(image) is not None:
                    # Resume the return-to-list wait from the confirmed result.
                    return False
                if any(self.recognition(node, image) for node in (
                        'IdolRoadFloorOverview', 'IdolRoadFormationPage', 'IdolRoadStartNextFlag')):
                    return True
                if repeated or time.monotonic() - start < hiro_module('runtime.tower_play').RESULT_REDO_S:
                    return None
                again = self.target('IdolRoadBackToList', image)
                if again is None:
                    return None
                if failed:
                    current = self.read_result_page(image) or self.read_unscored_failure(image)
                    same = current is not None and (current['rank'], current['score']) == (result.rank, result.score)
                else:
                    same = read_tower_clear_result(image, self.ocr(image), ocr=self.ocr, layer=layer) == (layer, stars)
                if same:
                    self.click(_center(again), purpose='tower-result-overview-repeat')
                    repeated = True
                return None

            if self.poll(departure, 'tower-next-page') is False:
                self.poll(departure, 'tower-next-page')
        if cache is not None and verdict == 'next':
            cache.finish()
            self._tower_result_cache = None
        return verdict

    def action_feedback(self, before, action, *, hiro_ui=False):
        drinks = tuple(pair for pair in before.drink_slots
                       if action.kind is not ActionKind.USE_DRINK or pair[0] != action.source_slot)
        previous_frame = self.live_screen.observation
        recovery = previous_frame.skip_recovery if action.kind is ActionKind.END_TURN else None
        turn_receipt = {'previous_frame': previous_frame} if action.kind is ActionKind.END_TURN else {}
        branches = {}
        next_animation = 0.
        next_dialog_click = 0.
        next_full_scan = 0.
        unread_since = None
        first_visible_successor = True
        confirmation = self.card_confirmation if action.kind is ActionKind.PLAY_CARD else None
        confirmation_retries = 0
        deselected = False

        def native_choice(image, *, panel=False):
            menu = self.native_session.pending_choice(before, action)
            if menu is None:
                return None
            # Hiro's ProduceExamEffectType numeric values for the known Maa menus.
            kind = {9: ActionKind.MOVE_CARD, 24: ActionKind.FREE_PLAY, 187: ActionKind.PAID_PLAY}.get(menu.effect_type)
            if kind is None:
                selector = _NATIVE_TARGET_EFFECTS.get(menu.effect_type)
                if selector is None or (not panel and not self.native_target_choice_visible(image, selector[0])):
                    return None
                kind, button = ActionKind.CHOOSE_TARGET, selector[1]
            else:
                if not panel and not self.choice_visible(image, kind):
                    return None
                button = _CHOICE_CONFIRMS[kind]
            from .idol_road_choice import execute_native_choice

            def confirm(frame):
                if kind is ActionKind.MOVE_CARD:
                    native = self.target('ProduceMoveCards', frame)
                    if native is not None:
                        return native
                buttons = battle_choice_buttons(frame)
                if buttons is not None:
                    return buttons[1]
                if kind is ActionKind.MOVE_CARD:
                    return None
                if kind is ActionKind.CHOOSE_TARGET:
                    return self.choice_text(frame, 'IdolRoadChoiceCommit', rf'.{{0,2}}(?:{button})', (365, 1105, 285, 150))
                return self.choice_text(frame, 'IdolRoadChoiceCommit', button, (200, 1105, 350, 150))

            receipt = execute_native_choice(self, image, menu, kind, confirm=confirm)
            self.native_session.record_choice(menu, receipt)
            self.log(dict(event='idol_road_native_choice_receipt', **vars(receipt)))
            self.poll(lambda frame: True if confirm(frame) is None else None, 'move-choice-follow-up')
            return True

        def projected(frame=None):
            key = (frame.turns, frame.attribute) if frame else None
            if key not in branches:
                branches[key] = tower_branches(self.kernel, before, action, self.multipliers, observed=frame,
                                               skip_recovery=recovery or 0, timeline=self.timeline)
            return branches[key]

        def read(image, *, after_timeout=False, choices_only=False):
            nonlocal next_animation, first_visible_successor, next_dialog_click, next_full_scan, unread_since
            nonlocal confirmation, confirmation_retries, deselected
            if choices_only:
                return self.native_session and native_choice(image, panel=battle_choice_buttons(image) is not None)
            retry = bool(self.recognition('IdolRoadRetry', image))
            if retry or self.result_page_ready(image):
                result = self.read_result_page(image)
                if result:
                    if result.get('score') is None:
                        # The exam boundary handles explicit unscored failure.
                        # A terminal screen alone cannot receipt a pending tap.
                        return None
                    self.retry_available = retry
                    if self.native_session:
                        receipt, modeled = self.native_session.observe_result(before, action, result)
                        self.log(dict(event='idol_road_maa_score', score=result['score'], modeled_score=modeled,
                                      difference=result['score'] - modeled if modeled is not None else None))
                        return receipt
                    try:
                        terminal = tower_terminal_step(before, projected(), self.frame_id, result['score'], drinks)
                    except (KernelError, HighScoreContractError):
                        terminal = None
                    if terminal:
                        step, modeled_score = terminal
                        self.log(dict(event='idol_road_maa_score', score=result['score'], modeled_score=modeled_score,
                                      difference=result['score'] - modeled_score))
                        return ExamResult(**result, terminal_step=step)
                    self.log(dict(event='idol_road_maa_score', score=result['score'],
                                  modeled_score=None, reason='game_ranking_without_modeled_terminal'))
                    return ExamResult(**result)
            if battle_choice_buttons(image) is not None:
                if after_timeout or hiro_ui:
                    return None
                if self.native_session:
                    native_choice(image, panel=True)
                    return None
                return self.read_choice(image, before, projected(), drinks, panel=True)
            if (hiro_ui and self.native_session and action.kind is ActionKind.PLAY_CARD and confirmation is None
                    and not self.native_session.choice_receipts):
                # A first tap alone can mean either a play or the game ending a turn.
                # Compare those explanations once instead of retrying as a confirmed play.
                return self.reobserve_pending(image, before, action, first_tap=True)
            now = time.monotonic()
            if not hiro_ui and not after_timeout and now >= next_dialog_click:
                slow = unread_since is not None and now - unread_since >= 5 and now >= next_full_scan
                if slow:
                    next_full_scan = now + 3
                dialog = self.pending_dialog(image, action, slow=slow)
                if dialog is not None:
                    self.click(_center(dialog), purpose='resolve-battle-dialog')
                    next_dialog_click = time.monotonic() + 1.
                    return None
            fresh = (self.observe_live(image, blocked_drink_slot=action.source_slot)
                     if action.kind is ActionKind.USE_DRINK else self.observe_live(image))
            if fresh is not None:
                unread_since = None
            elif unread_since is None:
                unread_since = now
            if (not hiro_ui and not after_timeout and confirmation is not None
                    and (confirmation_retries < 2 or self.native_session is not None)
                    and (fresh is None or self.same_live(fresh, self.live_screen))
                    and (selected := self.recheck_confirmed_card(image, confirmation)) is not None):
                if confirmation_retries < 2:
                    self.click(_center(selected), purpose='retry-confirm-card')
                    confirmation = replace(confirmation, sent_at=time.monotonic())
                    confirmation_retries += 1
                else:
                    self.click((360, 330), purpose='deselect-unplayable-card')
                    confirmation = None
                    deselected = True
                return None
            if deselected and fresh is not None:
                refreshed = self.reobserve_no_effect(before, action, fresh)
                if refreshed is not None:
                    self.unplayable_cards.add((before.state.turn, action.card_instance_id))
                    return refreshed
            if fresh is None:
                if after_timeout or hiro_ui:
                    return None
                if self.native_session:
                    if native_choice(image):
                        return None
                else:
                    choice = self.read_choice(image, before, projected(), drinks)
                    if choice is not None:
                        return choice
                ranking = before.state.turns_remaining == 1 and self.finished_rank_banner(image)
                if now >= next_animation and self.advance_exam_animation(image, ranking=ranking):
                    next_animation = time.monotonic() + .4
                return None
            if (after_timeout and action.kind is ActionKind.END_TURN and self.native_session
                    and fresh.observation.end_available):
                refreshed = self.reobserve_no_effect(before, action, fresh)
                if refreshed is not None:
                    return refreshed
            if before.decision_kind is DecisionKind.LIVE and self.same_live(fresh, self.live_screen):
                if after_timeout or (hiro_ui and self.native_session):
                    return None
                if self.native_session:
                    native_choice(image)
                    return None
                return self.read_choice(image, before, projected(), drinks)
            frame = fresh.observation
            if first_visible_successor:
                self.log(dict(event='idol_road_visible_successor', frame_id=frame.frame_id,
                              action=action.kind.value, turns=frame.turns, hand=frame.hand, drinks=frame.drinks,
                              score=frame.score, stamina=frame.stamina, genki=frame.genki))
                first_visible_successor = False
            if self.native_session:
                step = self.native_session.observe_frame(before, action, frame, **turn_receipt)
                if step is None:
                    forms = self.native_session.overlapping_hand_forms(before, action, frame)
                    corrected = self.read_overlapping_forms(image, fresh, forms)
                    if corrected != fresh:
                        fresh, frame = corrected, corrected.observation
                        step = self.native_session.observe_frame(before, action, frame, **turn_receipt)
                    if step is None and confirmation is not None:
                        step = self.native_session.observe_frame(before, action, frame, confirmed_card=True,
                                                                 previous_frame=previous_frame)
                if step:
                    if self.native_session.observed_delta:
                        self.log(dict(event='idol_road_maa_status_reconciled', frame_id=frame.frame_id,
                                      observed_minus_model=self.native_session.observed_delta))
                    self.accept_live_screen(fresh, step.request)
                return step
            candidates = projected(frame)
            step = tower_live_step(before, candidates, frame)
            if step is None:
                fresh = self.read_overlapping_hand(image, fresh, candidates)
                frame = fresh.observation
                step = tower_live_step(before, candidates, frame)
            if step is None:
                step = tower_observed_extra_action(self.kernel, before, action, frame)
                if step is not None:
                    self.log(dict(event='idol_road_observed_extra_action', frame_id=frame.frame_id))
            if step:
                corrected = {event.detail: event.amount for event in step.events
                             if event.event_type == 'TOWER_OBSERVED_VALUE'}
                if corrected:
                    self.log(dict(event='idol_road_maa_status_reconciled', frame_id=frame.frame_id,
                                  observed_minus_model=corrected))
                self.accept_live_screen(fresh, step.request)
                return step
            return None if after_timeout else self.read_choice(image, before, projected(), drinks)

        def refresh():
            if self.context.tasker.stopping:
                return None
            image = self.capture()
            if (observed := read(image, after_timeout=True)) is not None:
                return observed
            # Hiro keeps observing through the ending animation until the
            # result page or the exam stall limit. The usual action wait can
            # end while the result is still behind the cinematic.
            if (before.state.turns_remaining == 1 and (confirmation or action.kind is ActionKind.END_TURN)
                    and not self.recognition('ProduceRecognitionHealthFlag', image)
                    and _battle_dialog_header(image) is None):
                return self.poll(lambda frame: read(frame, after_timeout=True), 'tower-final-result',
                                 duration=EXAM_STALL_TIMEOUT)
            return None

        return ExamFeedback(read, refresh)

    def finished_rank_banner(self, image):
        rank = self.ocr(image, roi=(270, 827, 116, 140), expected=r'^[1-6]$', only_rec=True)
        if not any(row.score >= .95 and re.fullmatch(r'[1-6]', row.text) for row in rank):
            return False
        return any(row.text == '位' and row.score >= .95 for row in self.ocr(
            image, roi=(390, 906, 47, 57), expected='^位$', only_rec=True))

    def restore_tower_opening(self, algorithm, client_size, cache):
        """Rebind only a recorded opening that has no later Maa input."""
        if not cache.has_opening() or not cache.has_verified_formation():
            return False
        self.client_size, self.algorithm = client_size, algorithm
        self.prepare(algorithm, cached=cache.load(require_verified=True))
        fresh = self.read_tower_opening(self.capture())
        if fresh is None or not cache.opening_matches(fresh.observation):
            self.loadout = None
            return False
        self.live_screen = fresh
        self._tower_opening_cache = cache
        return True

    def save_committed_exam(self, request, frame):
        cache = getattr(self, '_tower_result_cache', None)
        if cache is None or self.native_session is None or getattr(self, '_saved_exam_frame', None) == frame.frame_id:
            return
        position = self.native_session.committed_position(request)
        if position is not None:
            try:
                cache.record_position(dict(position, algorithm=self.algorithm, frame=cache._opening_key(frame)))
            except OSError as error:
                self.log(dict(event='idol_road_committed_position_not_saved', reason=str(error)))
            else:
                self._saved_exam_frame = frame.frame_id

    def restore_tower_position(self, algorithm, client_size, cache):
        """Bind an idle, previously committed position to a fresh Maa frame."""
        if not cache.has_verified_formation() or (saved := cache.committed_position()) is None:
            return False
        try:
            self.client_size, self.algorithm = client_size, algorithm
            self.prepare(algorithm, cached=cache.load(require_verified=True))
            fresh = self.observe_live(self.capture())
            if (self.native_rules is not None and fresh is not None and saved['algorithm'] == self.algorithm
                    and saved['frame'] == cache._opening_key(fresh.observation)):
                frame = fresh.observation
                self.observed_types.clear()
                self._restored_exam_box, self.native_session = open_native_decision_box(
                    self.algorithm, self.native_rules, None, self.loadout, self.observed_types, item_uses={},
                    statuses=(), rust_stage=self.rust_stage, resumed=(saved, frame))
                self.live_screen = fresh
                self._tower_result_cache = cache
                self._saved_exam_frame = frame.frame_id
                return True
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            if isinstance(error, TowerDateChanged):
                raise
            self.log(dict(event='idol_road_committed_position_unavailable', reason=str(error)))
        cache.record_position(None)
        return False

    def run(self, algorithm, *, client_size, retry=None, attempts=None, saved=None):
        self.unplayable_cards.clear()
        self.unplayable_drinks.clear()
        self.observed_types.clear()
        self.native_session = None
        self.client_size = client_size
        self.algorithm = algorithm

        def check_attempts():
            nonlocal attempts
            if attempts is None:
                return
            try:
                allowed = attempts.allowed(self.loadout)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.log(dict(event='idol_road_daily_attempts_unread', reason=str(error)))
                attempts = None
                return
            if not allowed:
                raise ValueError('idol_road_daily_attempts_exhausted')

        retrying = retry is not None or (saved is not None and saved[1])
        read_opening = self.stable_tower_opening()
        if retrying:
            if retry is not None:
                self.__dict__.update(vars(retry))
            else:
                self.prepare(algorithm, cached=saved[0].load(require_verified=True))
            check_attempts()
            if saved is not None:
                saved[0].update_opening(None)
            self.enter_same_formation_retry()
            _, self.live_screen = self.poll(lambda image: self.after_retry(image, read_opening),
                                            'retry-opening-or-formation')
        if self.live_screen is None:
            if not retrying:
                if saved is None:
                    self.prepare(algorithm)
                else:
                    self.prepare(algorithm, remember=saved[0])
                check_attempts()
            target = self.poll(lambda image: self.target('IdolRoadStart', image), 'challenge')
            self.click(_center(target), purpose='challenge')
            self.live_screen = self.poll(read_opening, 'opening-hud')
        if saved is not None and getattr(self, '_tower_opening_cache', None) is None:
            saved[0].update_opening(self.live_screen.observation)
            self._tower_opening_cache = saved[0]
        if saved is not None:
            self._tower_result_cache = saved[0]
        return self.run_live_exam(attempts=attempts, hiro_label=f'tower-{self.loadout.overview.layer}'
                                  if getattr(self, 'hiro_policy_route', False) else None)

    def run_live_exam(self, *, attempts=None, exam=None, hiro_label=None, hiro_session=None):
        restored = getattr(self, '_restored_exam_box', None)
        if restored is not None:
            from .hiro_maa_exam import run_maa_exam

            self._restored_exam_box = None
            return run_maa_exam(self, restored, self.native_legal_reader(), label=hiro_label, session=hiro_session)
        items = self.opening_item_uses()
        if isinstance(self.kernel, TowerCardKernel):
            self.kernel.use_stage_timeline(self.timeline)
        frame = self.live_screen.observation
        self.observed_types.append(ATTRIBUTES.index(frame.attribute) + 1)
        unread = self.unread_formation_cards
        opening_deck = self.deck_ids
        if unread != 0:
            observed = tuple((Counter(frame.hand) - Counter(opening_deck)).elements())
            if unread is not None and len(observed) > unread:
                raise ValueError('idol_road_opening_exceeds_unread_formation_cards')
            opening_deck += observed
            if unread is not None:
                unread -= len(observed)
        opening = tower_opening(self.kernel, self.loadout, frame, opening_deck, item_uses=items,
                                timeline=self.timeline, native=self.native_rules is not None,
                                partial_deck=unread != 0)
        if self.native_rules:
            box, self.native_session = open_native_decision_box(self.algorithm, self.native_rules,
                opening, self.loadout, self.observed_types, item_uses=items, statuses=frame.statuses,
                status_slots=frame.status_slots, exam=exam, unknown_deck=unread, rust_stage=self.rust_stage)

            legal = self.native_legal_reader()
        else:
            box = open_idol_road_battle(self.algorithm, loadout=self.loadout, kernel=self.kernel, opening=opening,
                planning_seed=time.time_ns(), deadline=float('inf'), observed_types=self.observed_types,
                rust_stage=self.rust_stage)

            def legal(request):
                return tower_legal(self.kernel, request, self.live_screen.observation)

        if hiro_label is not None:
            from .hiro_maa_exam import run_maa_exam

            result = run_maa_exam(self, box, legal, label=hiro_label, session=hiro_session)
        else:
            result = run_exam(box, legal_actions=legal, execute=self.submit, feedback=self.action_feedback, poll=self.poll,
                              force_skip=True, on_forced_skip=lambda: self.log(dict(
                                  event='cultivation_exam_forced_skip' if exam is not None else 'idol_road_forced_skip')),
                              allow_unranked=bool(exam and exam.get('lesson')))
        if attempts is not None and result is not None and hiro_label is None:
            try:
                count = attempts.record(self.loadout, result)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.log(dict(event='idol_road_daily_attempt_not_saved', reason=str(error)))
            else:
                self.retry_available = self.retry_available and count < attempts.limit
                self.log(dict(event='idol_road_daily_attempt', layer=self.loadout.overview.layer,
                              count=count, limit=attempts.limit))
        return result

    def native_legal_reader(self):
        last_item_observation = None

        def legal(request):
            nonlocal last_item_observation
            observed = self.live_screen.observation
            if last_item_observation != request.snapshot_id:
                last_item_observation = request.snapshot_id
                changes = self.native_session.sync_item_uses(self.observed_item_uses(), status_slots=observed.status_slots)
                if changes:
                    self.log(dict(event='exam_item_uses_reconciled', frame_id=request.snapshot_id,
                                  observed_minus_model=changes))
            return tuple(action for action in self.native_session.legal_actions(request, observed)
                         if (request.state.turn, action.card_instance_id) not in self.unplayable_cards
                         and (action.kind is not ActionKind.USE_DRINK
                              or (request.state.turn, action.source_slot) not in self.unplayable_drinks))

        return legal
