"""Bind the pinned upstream UI machine to the existing audited Maa host.

This module supplies transport and receipts, never an action policy. Extracted
card selection/response methods remain upstream; selected-effect observation,
input authorization and callers' business recovery are explicit local adapters.
The revision label alone does not attest those adapters or their call sites.
"""
from __future__ import annotations

import time
from copy import deepcopy
from types import SimpleNamespace

from .hif_visual import hif_selection_click_box
from .hif_capture_adapter import _Forwarder, captured_at, _TaskerProxy
from .upstream_hif_card_ui import BattleRecognitionMixin

UPSTREAM_REVISION = 'fb8fd00cfcab031d9a968888ac2fa48b299b241b'

# HIF-only upstream patch 8a8d6541 adds the full-stamina SKIP icon.
HIF_SKIP_RECOGNITION = dict(recognition='TemplateMatch', roi=[590, 720, 130, 120],
    template=['produce/skip_round.png', 'produce/skip_round_1.png',
              'produce/skip_round_2.png', 'produce/skip_round_3.png'])


def unselected_art_footprints(slot_boxes, frame_shape):
    """Recover full footprints for an already unselected, calibrated art row."""
    count = len(slot_boxes)
    if frame_shape[:2] != (1280, 720) or count not in (4, 5):
        return None
    footprints = []
    for slot, (observed_slot, box) in enumerate(slot_boxes):
        anchor = round(18 + slot * 494 / (count - 1))
        x, y, width, height = box
        if (observed_slot != slot or not anchor - 8 <= x <= anchor + 190 - width + 8
                or not 878 <= y <= 904 or abs(y + height - 1134) > 16
                or not 90 <= width <= 206):
            raise ValueError('upstream fragment does not fit the calibrated unselected art row')
        footprints.append((anchor, 884, 190, 250))
    return tuple(footprints)


def upstream_saved_navigation_target(context, image, node):
    """The two upstream continue buttons; never a new-produce/start control."""
    templates = {'ProduceContinue': 'home_1.png',
        'ProduceChooseContinue': 'produce/continue.png'}
    if node not in templates:
        raise ValueError('unsupported upstream saved-game navigation node')
    result = context.run_recognition(node, image, pipeline_override={node: dict(
        recognition='TemplateMatch', template=[templates[node]])})
    if not result or not result.hit:
        raise ValueError('upstream saved-game navigation target is absent: ' + node)
    return list(result.best_result.box)

# Recognition parameters from that revision's ProduceUtils.json, Produce.json
# and battle_recognition._hif_has_select (the dynamic SELECT node).
# Namespace removal and flat pipeline syntax are Maa transport adaptation only.
_NODES = {
    'ProduceRecognitionCards': dict(recognition='NeuralNetworkDetect', expected=[0, 1, 2],
        model='cards.onnx', labels=['cards', 'suggestions', 'useless']),
    'ProduceRecognitionSkipRound': HIF_SKIP_RECOGNITION,
    'ProduceRecognitionHealthFlag': dict(recognition='TemplateMatch', roi=[512, 125, 168, 130],
        template='produce/health.png', green_mask=True, threshold=.7),
    'ProduceButton': dict(recognition='TemplateMatch', roi=[0, 780, 720, 500],
        template=['next.png', 'produce/round_close_button.png', 'produce/retry.png', 'produce/yes.png',
                  'produce/start.png', 'produce/close.png', 'produce/shop_exit.png', 'cancel.png'], order_by='Score'),
    'ProduceIdentitySelected': dict(recognition='OCR', expected='SELECT'),
    'ProduceIdentityDetail': dict(recognition='OCR', expected=r'[^\n]+'),
}


class _Capture(_Forwarder):
    def __init__(self, job, owner):
        super().__init__(job)
        self._hif_owner = owner

    def wait(self):
        self._hif_wrapped.wait()
        return self

    def get(self, *args, **kwargs):
        image = self._hif_wrapped.get(*args, **kwargs)
        owner = self._hif_owner
        owner._previous_frame_id = owner.frame_id
        owner.image = image
        owner.frame_time = captured_at(owner.base.tasker.controller, image)
        owner.frame_id = owner.frame_identifier(image)
        return image


class _Controller(_Forwarder):
    def __init__(self, controller, owner):
        super().__init__(controller)
        self._hif_owner = owner

    def post_screencap(self):
        return _Capture(self._hif_wrapped.post_screencap(), self._hif_owner)

    def post_click(self, x, y):
        owner = self._hif_owner
        # Only the user-approved strong selected-card submission has 2 s.
        maximum_age = 2. if owner.phase == 'confirm' else 1.
        floor = owner.frame_time
        if floor is None:
            raise ValueError('upstream UI has no current frame')
        deadline = floor + maximum_age
        row = dict(phase=owner.phase, x=x, y=y, frame_id=owner.frame_id,
            captured_at=floor, input_deadline=deadline, dispatched=False,
            dispatch_index=owner.base.gkh_hif_upstream_dispatch_count())
        owner.receipt['inputs'].append(row)
        owner.publish()

        def before_dispatch():
            if owner.base.tasker.stopping:
                raise ValueError('upstream UI owner is stopping')
            now = time.perf_counter()
            if not floor <= now <= deadline:
                raise ValueError('upstream UI input evidence expired')
            return now

        job = owner.base.gkh_hif_post_live_click(x, y, before_dispatch=before_dispatch,
            input_deadline=deadline, authorization_floor=floor,
            preparation_key='native_upstream_input_preparation')
        row.update(dispatched=True, job_id=job.job_id)
        owner.publish()
        return job


def upstream_drink_actions(context, image):
    """fb8 battle_drink._find_drink_detail_actions, on the current Maa frame."""
    score = 'ProduceRecognitionScore'
    buttons = dict(recognition='OCR', roi=[80, 1120, 550, 140], expected='使う|キャンセル|キヤンセル')
    try:
        match = context.run_recognition(score, image, pipeline_override={score: buttons})
    except Exception:
        return {}
    if not (match and match.hit and match.filtered_results):
        return {}
    actions = {}
    for row in match.filtered_results:
        label = (row.text or '').replace(' ', '').replace('　', '')
        if '使う' in label:
            actions['use'] = tuple(row.box)
        elif 'キャンセル' in label or 'キヤンセル' in label:
            actions['cancel'] = tuple(row.box)
    return actions


def upstream_drink_detail_open(context, image):
    return len(upstream_drink_actions(context, image)) == 2


def wait_after_confirmed_move(context):
    """fb8 battle_drink._wait_for_battle_drink_return, after its one confirmed move."""
    deadline = time.monotonic() + 4
    stable = 0
    poll = getattr(context, 'gkh_hif_poll_battle_progress', None)
    while not context.tasker.stopping:
        image = context.tasker.controller.post_screencap().wait().get()
        move = context.run_recognition('ProduceRecognitionChooseMoveCards', image)
        if move and move.hit:
            stable = 0
        else:
            skip = 'ProduceRecognitionSkipRound'
            playable = context.run_recognition(skip, image, pipeline_override={skip: _NODES[skip]})
            stable = stable + 1 if playable and playable.hit and not upstream_drink_detail_open(context, image) else 0
            if stable == 2:
                return
        if time.monotonic() >= deadline:
            if not callable(poll) or not (move and move.hit or stable):
                return
            # The original move was delivered. Stay on its visible page until
            # it changes or the host's existing no-progress recovery concludes.
            poll()
        time.sleep(.2)
    # The caller's normal observation still checks the actual outcome. It has
    # the original confirmation receipt and cannot repeat the move or the drink.


def _separate_detail_cost_icon(image, rows, titles):
    """Keep complete title fragments and exclude only the independent cost icon."""
    if titles is None or len(rows) < 2 or titles.lookup(''.join(row.text or '' for row in rows))[0]:
        return rows, None
    from .hif_visual import locate_hif_selected_detail_title
    from .hif_skill_support import _panel_bounds, detail_cost_icon_box, detail_row_within_region

    title_roi = locate_hif_selected_detail_title(image)
    if title_roi is None:
        return rows, None
    try:
        panel = _panel_bounds(image, title_roi)
    except ValueError:
        return rows, None
    icon = detail_cost_icon_box(image, panel)
    if icon is None:
        return rows, None
    name = [row for row in rows if row.box[0]+row.box[2] <= icon[0]]
    if (not name or len(name) == len(rows) or not titles.lookup(''.join(row.text or '' for row in name))[0]
            or any(not detail_row_within_region(image, row.box, icon) for row in rows[len(name):])):
        return rows, None
    return name, icon


class _Context(_Forwarder):
    def __init__(self, owner):
        super().__init__(owner.base)
        self._hif_owner = owner
        self._hif_tasker = _TaskerProxy(owner.base.tasker, _Controller(owner.base.tasker.controller, owner))

    @property
    def tasker(self):
        return self._hif_tasker

    def run_recognition(self, node, image, *, pipeline_override=None, **kwargs):
        name = node.removeprefix('ProduceHIF__')
        if name not in _NODES:
            raise ValueError('unknown upstream UI recognition node: ' + node)
        override = deepcopy(_NODES[name])
        if pipeline_override:
            override.update(pipeline_override[node])
        result = self._hif_wrapped.run_recognition(name, image,
            pipeline_override={name: override}, **kwargs)
        owner = self._hif_owner
        if name == 'ProduceRecognitionSkipRound':
            hit = bool(result and result.hit)
            owner.receipt['skip_frames'].append(dict(frame_id=owner.frame_identifier(image), hit=hit))
        if name == 'ProduceIdentityDetail':
            rows = sorted(result.filtered_results, key=lambda row:row.box[0]) if result and result.hit else []
            x, y, w, h = override['roi']
            texts = [dict(text=row.text or '',score=float(row.score),
                box=[row.box[0]-x,row.box[1]-y,*row.box[2:]]) for row in rows]
            observation = dict(frame_id=owner.frame_identifier(image), roi=[x,y,w,h], ocr=texts)
            title_rows, icon = _separate_detail_cost_icon(image, rows, getattr(owner, 'inspection_titles', None))
            if icon is not None:
                observation.update(raw_ocr=texts, ocr=texts[:len(title_rows)], excluded_cost_icon=icon)
                result = SimpleNamespace(hit=True, filtered_results=title_rows)
            owner.receipt.setdefault('detail_observations', []).append(observation)
        return result

    def run_task(self, node):
        # Upstream runs these recognition+Click nodes. Route the same target
        # through the existing endpoint/deadline audit, not an unaudited task.
        name = node.removeprefix('ProduceHIF__')
        if name not in ('ProduceButton', 'ProduceRecognitionSkipRound'):
            raise ValueError('unknown upstream UI action node: ' + node)
        image = self.tasker.controller.post_screencap().wait().get()
        result = self.run_recognition(node, image)
        if not result or not result.hit:
            return SimpleNamespace(succeeded=False)
        box = result.best_result.box
        x, y = box[0] + box[2] // 2, box[1] + box[3] // 2
        if name == 'ProduceButton':
            y -= 5  # Upstream target_offset [0,0,0,-10] changes box height.
        old = self._hif_owner.phase
        self._hif_owner.phase = 'navigation' if name == 'ProduceButton' else 'end_turn'
        try:
            job = self.tasker.controller.post_click(x, y).wait()
            if name == 'ProduceRecognitionSkipRound':
                time.sleep(.5)
            return job
        finally:
            self._hif_owner.phase = old


class UpstreamHifCardUI(BattleRecognitionMixin):
    """A single action's upstream machine, with no reusable input credentials."""

    def __init__(self, context, *, frame_identifier, sink, move_cards=None, selected=None):
        self.base = context
        self.frame_identifier = frame_identifier
        self.sink = sink
        self.move_cards = move_cards
        self.selected = selected
        self.phase = 'select'
        self.image = self.frame_time = self.frame_id = None
        self._previous_frame_id = None
        self.selected_frames = []
        self._initial_play_row = None
        self._click_frame = self._first_click = None
        self._refreshing_selection = False
        self.receipt = dict(kind='upstream_hif_card_ui', revision=UPSTREAM_REVISION,
            inputs=[], skip_frames=[], response_confirmed=False, playable=False)
        self.context = _Context(self)
        self._hif_debug_enabled = False
        self._last_turn_count = None
        self._reset_hif_recognition()

    def publish(self):
        self.sink(deepcopy(self.receipt))

    def _hif_log_changed(self, *_args, **_kwargs):
        pass

    def _hif_log_reset(self, *_args):
        pass

    def _hif_log_selection_pair(self, *_args):
        pass

    @staticmethod
    def _card_label(key):
        return str(key)

    @staticmethod
    def _hif_card_name(text):
        from .hif_single_plus import normalize_hif_title

        return normalize_hif_title(text)

    def _hif_name_key(self, text):
        card_id, self.inspected_family = self.inspection_titles.lookup(text)
        return card_id

    def _handle_move_cards(self, context, image):
        return bool(self.move_cards(context, image)) if self.move_cards else False

    def _hif_selected_index(self, context, image, boxes, *, pending=True):
        selected = super()._hif_selected_index(context, image, boxes, pending=pending)
        self._initial_play_row = None
        if (pending and selected == -1 and len(boxes) == 5 and not self.receipt['inputs']
                and self.receipt.get('action', {}).get('kind') == 'PLAY_CARD'):
            self._initial_play_row = [box for box, _ in boxes], image.shape
        return selected

    def _hif_frame_boxes(self, context, image):
        from .exam_hand import anchor_exam_hand, playable_exam_boxes

        recognition = context.run_recognition('ProduceHIF__ProduceRecognitionCards', image)
        rows = recognition.all_results if recognition else ()
        anchored = anchor_exam_hand(image, rows, self)
        boxes = self._hif_card_boxes(rows) if anchored is None else playable_exam_boxes(anchored)
        if image is self.image and self.receipt.get('action', {}).get('kind') == 'PLAY_CARD':
            self._click_frame = image, boxes
        return boxes

    def _retry_hand_signature(self, box, index):
        image = self.image
        if (getattr(image, 'shape', ()) != (1280, 720, 3)
                or self._click_frame is None or self._click_frame[0] is not image):
            return None
        boxes = self._click_frame[1]
        if (not self._hif_same_geometry(boxes) or not 0 <= index < len(boxes)
                or boxes[index][0] != box):
            return None
        panel = self._hif_panel_top(image)
        detail = None
        if self.phase == 'confirm':
            if panel is None:
                return None
            detail = panel, image[panel + 8:panel + 48, 150:560].tobytes()
        elif panel is not None or any(b[1] <= 875 and b[1] + b[3] <= 1130 for b, _ in boxes):
            return None
        # Include the art and title band, including any non-playable slots
        # omitted by upstream's playable-box detector.
        return tuple(tuple(b) for b, _ in boxes), image[840:1170].tobytes(), detail

    def _retry_click_box(self, context, box, index, action):
        first, self._first_click = self._first_click, None
        inputs = self.receipt['inputs']
        skip = self.receipt['skip_frames']
        if (first is None or (first.action, first.phase, first.index) != (action, self.phase, index)
                or not inputs or not inputs[-1]['dispatched'] or inputs[-1]['frame_id'] != first.frame_id
                or inputs[-1]['phase'] != self.phase or self.frame_id in (None, first.frame_id, self._previous_frame_id)
                or self.frame_time is None or self.frame_time <= inputs[-1]['captured_at']
                or not skip or skip[-1] != dict(frame_id=self.frame_id, hit=True)
                or first.hand is None or self._retry_hand_signature(box, index) != first.hand):
            return None

        # Only a retry reads the entire marker band to rule out a second or
        # partial SELECT. Leave upstream's ordinary per-card OCR untouched.
        node = 'ProduceHIF__ProduceIdentitySelected'
        result = context.run_recognition(node, self.image,
            pipeline_override={node: {'roi': [0, 1106, 720, 40]}})
        if result is None:
            return None
        if self.phase == 'select':
            return first.click_box if not result.hit else None
        rows = result.filtered_results if result.hit else ()
        if len(rows) != 1 or rows[0].text != 'SELECT':
            return None
        x, _, width, _ = rows[0].box
        center = x + width / 2
        slots = [i for i, (left, _, w, _) in enumerate(first.hand[0])
                 if abs(left + w // 2 - center) <= 70]
        return first.click_box if slots == [index] else None

    def _hif_click_card(self, context, box, index, phase, attempt):
        self.phase = 'confirm' if phase == '确认出牌' else 'select'
        action = self.receipt.get('action', {})
        pixel_retry = action.get('kind') == 'PLAY_CARD' and self.receipt.get('confirmation_mode') != 'observe_next_page'
        if pixel_retry and self._refreshing_selection:
            return False
        if pixel_retry and attempt:
            retry_box = self._retry_click_box(context, box, index, action)
            if retry_box is None:
                self.receipt['retry_blocked'] = dict(phase=self.phase, frame_id=self.frame_id,
                    reason='current_hand_not_proven_unchanged')
                self.publish()
                return False
            return super()._hif_click_card(context, retry_box, index, phase, attempt)

        detector_box = box
        row, self._initial_play_row = self._initial_play_row, None
        if phase == '选中' and attempt == 0 and row is not None:
            boxes, frame_shape = row
            if 0 <= index < len(boxes) and boxes[index] == box:
                try:
                    footprints = unselected_art_footprints(list(enumerate(boxes)), frame_shape)
                    if footprints is not None:
                        box = hif_selection_click_box(footprints, index)
                except ValueError:
                    pass  # Other geometries retain the upstream detector target.
        if pixel_retry:
            self._first_click = SimpleNamespace(action=deepcopy(action), phase=self.phase, index=index,
                frame_id=self.frame_id, click_box=tuple(box), hand=self._retry_hand_signature(detector_box, index))
        return super()._hif_click_card(context, box, index, phase, attempt)

    def _hif_select_card(self, context, image, index):
        image = super()._hif_select_card(context, image, index)
        if image is not None and self.selected is not None:
            self.selected(context, image)
            if self.receipt.get('selected_skill_support') is not None:
                # Detail OCR is computation, not a fresh input certificate.
                # Reuse upstream's already-selected path; it sends no new
                # selection and returns its normal two-frame confirmation.
                self._refreshing_selection = True
                try:
                    image = super()._hif_select_card(context, image, index)
                finally:
                    self._refreshing_selection = False
                self.receipt['detail_refreshed_for_input'] = image is not None
                self.receipt['refreshed_detail_frames'] = [key for key, _ in self.selected_frames]
                self.publish()
        return image

    def _hif_has_select(self, context, image, box):
        found = super()._hif_has_select(context, image, box)
        if found and self._hif_panel_top(image) is not None:
            key = self.frame_identifier(image)
            if not self.selected_frames or self.selected_frames[-1][0] != key:
                self.selected_frames.append((key, image))
                self.selected_frames = self.selected_frames[-2:]
        else:
            self.selected_frames = []
        return found

    def _hif_selection_visible(self, context, image, boxes):
        visible = super()._hif_selection_visible(context, image, boxes)
        if self.phase == 'cancel_selection':
            self.receipt.setdefault('clear_observations', []).append(dict(
                frame_id=self.frame_identifier(image), selected=visible, has_cards=bool(boxes)))
        return visible

    def clear_selection(self):
        """Use miyu's neutral cancel and its consecutive unselected observations."""
        self.phase = 'cancel_selection'
        self.receipt['cleared'] = False
        try:
            self.receipt['cleared'] = super()._hif_clear_selection(self.context)
            return self.receipt['cleared']
        finally:
            self.publish()

    def _wait_until_playable(self, context, confirmation_count=1):
        # The upstream caller reaches this after its two-frame UI response.
        self.receipt['response_confirmed'] = True
        self.publish()
        ready = super()._wait_until_playable(context, confirmation_count)
        self.receipt['playable'] = ready
        self.publish()
        return ready

    def play(self, *, box, hand_boxes, action):
        self.selected_frames = []
        self._first_click = None
        self.receipt.pop('retry_blocked', None)
        self.receipt.update(action=deepcopy(action), target_box=list(box))
        self._hif_bind_hand(hand_boxes)
        self.publish()
        try:
            confirmed = self._play_a_card(self.context, list(box))
            self.receipt['response_confirmed'] = confirmed
            return confirmed
        finally:
            self._first_click = self._click_frame = None
            self.publish()

    def _first_target_select(self, index):
        """Use hiro's target marker and one same-scene retry on the bound miyu slot."""
        from .hif_card_retry import same_retry_scene, wait_for_target_select, read_exam_select_markers

        context = self.context
        resumed = getattr(self.base, 'gkh_hif_reobserved_selected_target', None)
        resume_selected = isinstance(resumed, dict) and resumed.get('action') == self.receipt['action']
        if resume_selected:
            self.base.gkh_hif_reobserved_selected_target = None
            self.receipt['selection_retry'] = dict(resumed.get('selection_retry') or {},
                outcome='observe_current_page', reobserved_frame_id=resumed['frame_id'],
                marker_observed=resumed.get('marker_observed') is True)

        def target_marker(image, boxes):
            current = self._hif_preview_box(boxes, index)
            box = current if current is not None else self._hif_hand_boxes[index]
            if BattleRecognitionMixin._hif_has_select(context, image, box):
                return box, True
            markers, present = read_exam_select_markers(context, image)
            target_x = box[0] + box[2] / 2
            for left, _, width, _ in markers:
                if abs(left + width / 2 - target_x) <= 70:
                    return box, True
            return None, present

        image = context.tasker.controller.post_screencap().wait().get()
        boxes = self._hif_frame_boxes(context, image)
        target = self._hif_preview_box(boxes, index)
        if resume_selected:
            return None if context.tasker.stopping else target_marker(image, boxes)[0]
        if context.tasker.stopping:
            return None
        selected = self._hif_selected_index(context, image, boxes)
        if selected == index:
            return target if target is not None else self._hif_hand_boxes[index]
        if selected == -2 or target is None:
            return target_marker(image, boxes)[0]
        skip = context.run_recognition('ProduceHIF__ProduceRecognitionSkipRound', image)
        if not (skip and skip.hit):
            return None
        first_frame = image
        if not self._hif_click_card(context, target, index, '选中', 0):
            inputs = self.receipt['inputs']
            if inputs and inputs[-1]['dispatch_index'] < self.base.gkh_hif_upstream_dispatch_count():
                self.receipt['selection_retry'] = dict(first_frame_id=inputs[-1]['frame_id'],
                    retries=0, outcome='observe_current_page', reason='uncertain_selection_dispatch')
            return None
        first_click = self.receipt['inputs'][-1]
        self._hif_selection_index = index
        self._hif_unselected_count = 0
        def capture():
            return context.tasker.controller.post_screencap().wait().get()

        for attempt in range(2):
            marker_box, saw_other_marker = wait_for_target_select(capture,
                lambda image: target_marker(image, self._hif_frame_boxes(context, image)),
                lambda: context.tasker.stopping)
            if marker_box is not None:
                return marker_box
            retry = self.receipt.setdefault('selection_retry', dict(first_frame_id=first_click['frame_id'],
                x=first_click['x'], y=first_click['y'], retries=0))
            if context.tasker.stopping or saw_other_marker or attempt:
                retry.update(outcome='observe_current_page', saw_other_marker=saw_other_marker)
                return None
            before = self.read_retry_scene(first_frame)
            image = context.tasker.controller.post_screencap().wait().get()
            boxes = self._hif_frame_boxes(context, image)
            marker_box, any_marker = target_marker(image, boxes)
            if marker_box is not None:
                return marker_box
            after = self.read_retry_scene(image)
            retry.update(before=before, after=after, retry_frame_id=self.frame_id)
            if not same_retry_scene(before, after) or any_marker:
                retry['outcome'] = 'observe_current_page'
                return None
            retry.update(outcome='retry_same_scene', retries=1)
            original_center = [first_click['x'], first_click['y'], 0, 0]
            if not super()._hif_click_card(context, original_center, index, '选中', 1):
                retry['outcome'] = 'observe_current_page'
                return None
        return None

    def submit_card_once(self, *, index, hand_boxes, action):
        """Confirm once on the target marker and return to the battle observer."""
        self.selected_frames = []
        self._first_click = None
        self.receipt.update(action=deepcopy(action), target_box=list(hand_boxes[index][0]),
                            confirmation_mode='observe_next_page')
        self._hif_bind_hand(hand_boxes)
        self.publish()
        try:
            target = self._first_target_select(index)
            if target is None:
                if self.receipt.get('selection_retry') is not None:
                    self.receipt['selection_retry'].update(bound_index=index,
                        bound_boxes=[list(box) for box, _ in hand_boxes])
                return False
            selected = self.image
            self.selected_frames = [(self.frame_identifier(selected), selected)]
            self.receipt['selected_target_frame'] = self.selected_frames[0][0]
            dispatched = self._hif_click_card(self.context, target, index, '确认出牌', 0)
            if self.selected is not None:
                try:
                    self.selected(self.context, selected)
                except ValueError as error:
                    self.receipt['selected_support_observation'] = dict(frame_id=self.selected_frames[0][0],
                        verified=False, reason=str(error))
            return dispatched
        finally:
            self._first_click = self._click_frame = None
            self.publish()

    def retry_persistent_confirmation(self, index):
        """Recheck the target marker after an unresolved submission of this action."""
        from .exam_hud import battle_dialog_header
        from .hif_card_retry import same_retry_scene, read_exam_select_markers, recheck_persistent_select

        context = self.context
        if context.tasker.stopping or not self.selected_frames:
            return False
        original = self.selected_frames[0][1]
        scene = self.read_retry_scene(original)
        if scene is None:
            return False

        def capture():
            return context.tasker.controller.post_screencap().wait().get()

        def selected(image, box):
            if battle_dialog_header(image) is not None:
                return None
            current = self._hif_preview_box(self._hif_frame_boxes(context, image), index)
            if current is None:
                return None
            x, y, w, h = current
            left, top, width, height = box
            if abs(x + w / 2 - left - width / 2) > 30 or abs(y + h - top - height) > 30:
                return None
            markers, _ = read_exam_select_markers(context, image)
            if (any(abs(mx + mw / 2 - x - w / 2) <= 70 for mx, _, mw, _ in markers)
                    and same_retry_scene(scene, self.read_retry_scene(image))):
                return current
            return None

        box = self._hif_preview_box(self._hif_frame_boxes(context, original), index)
        if box is None:
            return False
        checked = recheck_persistent_select(capture(), box, selected=selected, capture=capture,
                                            stopping=lambda: context.tasker.stopping)
        if checked is None:
            return False
        return self._hif_click_card(context, checked, index, '确认出牌', 0)

    def end_turn(self, *, action):
        """The SKIP recognition/Click used by pinned fb8 _skip_round.

        Its main-loop waiting/navigation is owned here by the existing complete
        rule settlement observer, not a second concurrent battle loop.
        """
        self.receipt['action'] = deepcopy(action)
        self.publish()
        try:
            return bool(self.context.run_task('ProduceHIF__ProduceRecognitionSkipRound').succeeded)
        finally:
            self.publish()

    def deterministic_confirmation(self, action):
        """Describe the upstream response; do not claim a reread hand removal."""
        frames = self.receipt['skip_frames'][-3:]
        if (self.receipt['response_confirmed'] is not True or self.receipt['playable'] is not True
                or len(frames) != 3 or not all(row['hit'] for row in frames)
                or len({row['frame_id'] for row in frames}) != 3):
            return None
        result = dict(kind='upstream_ui_response', revision=UPSTREAM_REVISION,
            action_instance_id=action.card_instance_id, frame_ids=[row['frame_id'] for row in frames],
            response_confirmed=True, playable=True)
        if self.receipt.get('selected_skill_support') is not None:
            result['selected_skill_support'] = deepcopy(self.receipt['selected_skill_support'])
        return result
