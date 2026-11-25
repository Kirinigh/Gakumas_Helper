"""Bridge Hiro's exam driver to Maa perception, dispatch and the strategy factory."""

import time
from types import SimpleNamespace
from contextlib import contextmanager
from dataclasses import replace

from .upgrade import compact_ocr_text
from .exam_hud import battle_dialog_header, battle_choice_buttons
from .exam_loop import ExamResult, ExamNoProgress, ExamReobserved, ExamSettlement, ExamCardConfirmation
from .hiro_runtime import hiro_module, hiro_binding
from .hif_card_retry import read_exam_select_markers
from .hif_upstream_ui import upstream_drink_actions
from .hiro_maa_device import hiro_maa_ocr
from .high_score.contracts import ActionKind, DecisionKind


@contextmanager
def _perception(module, **providers):
    original = {name: getattr(module, name) for name in providers}
    vars(module).update(providers)
    try:
        yield
    finally:
        vars(module).update(original)


def _key(action):
    if action.kind is ActionKind.PLAY_CARD:
        return f'play:{action.source_slot}'
    if action.kind is ActionKind.USE_DRINK:
        return f'drink:{action.source_slot}'
    return 'skip' if action.kind is ActionKind.END_TURN else None


def communication_frame(host, image, *, retry_only=False):
    """Supply Hiro's interrupt handler with Maa's dialog and OCR coordinates."""
    header = battle_dialog_header(image)
    if header is None:
        return SimpleNamespace(img=image, has_dialog=False, boxes=(), dialog=None)
    top, bottom = header
    rows = host.ocr(image, roi=(0, top, image.shape[1], image.shape[0] - top))
    comm = hiro_module('loop.comm_error')
    boxes = []
    title = None
    for row in rows:
        x, y, w, h = row.box
        text = compact_ocr_text(row.text)
        if top <= y + h // 2 <= bottom and any(hint in text for hint in (
                *comm.TITLE_HINTS, '通信中にエラー', '网络错误', '通信错误')):
            text = title = '通信エラー'
        elif text in ('重试', '重試'):
            text = comm.RETRY
        boxes.append((text, row.score, (x, y, x + w, y + h)))
    current = comm.comm_error(boxes) if title else None
    if current is None or (retry_only and current['kind'] != 'retry'):
        return SimpleNamespace(img=image, has_dialog=False, boxes=(), dialog=None)
    return SimpleNamespace(img=image, has_dialog=True, boxes=boxes, dialog=dict(kind='comm_error', title=title, y=top))


class ExamCommunication:
    def __init__(self, *, capture, ocr, tap, retry_only=False):
        self.ocr, self.retry_only = ocr, retry_only
        self.runtime = SimpleNamespace(screen=SimpleNamespace(shot=lambda: self.frame(capture())),
            device=SimpleNamespace(tap=tap), clock=time.monotonic, sleep=time.sleep, session=None)

    def frame(self, image):
        return communication_frame(self, image, retry_only=self.retry_only)

    def recover(self, image):
        frame = self.frame(image)
        return hiro_module('runtime.interrupts').comm(frame, self.runtime) if frame.has_dialog else None


def recover_communication(host, image):
    recovery = getattr(host, '_hiro_communication', None)
    if recovery is None:
        # Some callers do not expose OCR unless an interrupt is actually visible.
        recovery = host._hiro_communication = ExamCommunication(capture=host.capture,
            ocr=lambda frame, **kwargs: host.ocr(frame, **kwargs),
            tap=lambda x, y: host.click((x, y), purpose='hiro-network-retry'))
    detail = recovery.recover(image)
    if detail is None:
        return False
    host.log(dict(event='hiro_communication', detail=detail))
    return True


class MaaExam:
    def __init__(self, exam, host, box, legal):
        self.exam, self.host, self.box, self.legal = exam, host, box, legal
        self.frame = None
        self.feedback = None
        self.settlement = None
        self.action_taps = 0
        self.no_effect = False
        self.waiting_since = None
        self.action_applied = False
        self.executing = False
        self.drink_point = None
        self.last_job = None
        self.reconciling = False

    def rect(self, box):
        x, y, w, h = box
        return self.exam.Box(x, y, x + w, y + h)

    def screenshot(self):
        self.checkpoint()
        return SimpleNamespace(img=self.host.capture())

    def checkpoint(self):
        if self.host.context.tasker.stopping:
            raise ExamNoProgress('maa_exam_stopped')

    def recognize(self, image):
        if self.frame is not None and self.frame.img is image:
            return self.frame
        self.frame = self.recognized_frame(image, self.host.observe_live(image))
        return self.frame

    def recognized_frame(self, image, screen):
        hud = screen.observation if screen is not None else None
        cards = [SimpleNamespace(rect=self.rect(box), card=SimpleNamespace(name=cid),
                                available=slot in hud.playable)
                 for slot, (box, cid) in enumerate(zip(screen.boxes, hud.hand, strict=True))] if hud else []
        drinks = [(SimpleNamespace(id=f'{slot}:{cid}', name=f'{slot}:{cid}'),
                   (*screen.drinks[slot]['center'], 1, 1)) for slot, cid in hud.drinks
                  if slot in screen.drinks] if hud else []
        return MaaExamFrame(image, screen, cards, drinks)

    def signature(self, image):
        screen = self.frame.screen if self.frame is not None and self.frame.img is image else None
        return self.host.read_progress_signature(image, screen=screen)

    def texts(self, image, y1, y2):
        if (y1, y2) == self.exam.SELECT_BAND:
            marks, _ = read_exam_select_markers(self.host.context, image)
            return [self.exam.Text('SELECT', self.rect(box)) for box in marks]
        return [self.exam.Text(row.text, self.rect(row.box), row.score)
                for row in self.host.ocr(image, roi=(0, y1, image.shape[1], min(y2, image.shape[0]) - y1))]

    def dialog_top(self, image):
        header = battle_dialog_header(image)
        if header is not None:
            return header[0]
        return 100 if battle_choice_buttons(image) is not None else None

    def drink_detail(self, image):
        actions = upstream_drink_actions(self.host.context, image)
        use = actions.get('use')
        return bool(actions) or self.dialog_top(image) is not None, (
            self.exam.Text('使う', self.rect(use)) if use is not None else None)

    def dialogs(self, image=None, *, slow=False, select=True):
        image = self.screenshot().img if image is None else image
        if recover_communication(self.host, image):
            return True
        if self.feedback is not None and self.action_taps and (self.dialog_top(image) is not None or slow):
            if self.feedback.read(image, choices_only=True):
                return True
            if battle_choice_buttons(image) is not None:
                if self.observe(image):
                    return True
                return False
        handled = self.exam._exam_dialogs(image, slow=slow, select=select)
        action = self.box.pending_action
        if not handled and self.action_taps and action is not None:
            target = self.host.pending_dialog(image, action, slow=slow)
            if target is not None:
                self.tap(*self.rect(target).center)
                return True
        return handled

    def execute(self, upstream, decision, frame):
        action = self.box.pending_action
        self.last_job = None
        if action.kind is ActionKind.PLAY_CARD:
            self.exam.PLAY_RESULT['confirmed'] = False
            image = self.screenshot().img
            target = self.host.current_card_target(image, self.host.live_screen, action.source_slot)
            if target is None:
                return []
            fresh, slot, _ = target
            frame = self.recognized_frame(image, fresh)
            self.frame = frame
            if slot != decision.slot:
                decision = replace(decision, action=f'play:{slot}')
        elif action.kind is ActionKind.USE_DRINK:
            self.drink_point = tuple(frame.drinks[decision.slot][1][:2])
        self.executing = True
        try:
            taps = upstream(decision, frame)
        finally:
            self.executing = False
            self.drink_point = None
        if action.kind is ActionKind.PLAY_CARD and self.exam.PLAY_RESULT.get('confirmed'):
            self.confirm(frame.cards[decision.slot].rect.xywh)
        elif self.action_taps:
            self.feedback = self.host.action_feedback(self.box.request, action, hiro_ui=True)
        return taps

    def confirm(self, rect):
        self.waiting_since = time.monotonic()
        self.host.card_confirmation = ExamCardConfirmation(rect, self.waiting_since)
        self.feedback = self.host.action_feedback(self.box.request, self.box.pending_action, hiro_ui=True)

    def persistent_select(self, upstream, marker):
        action = self.box.pending_action
        job = self.last_job
        handled = upstream(marker)
        if (handled and self.action_taps and action is not None and action.kind is ActionKind.PLAY_CARD
                and self.last_job != job and not self.exam.UNPLAYABLE.get('hit')):
            self.confirm(marker.rect.xywh)
        if handled and self.exam.UNPLAYABLE.get('hit'):
            self.no_effect = True
        return handled

    def wait_change(self, upstream, signature):
        self.waiting_since = time.monotonic()
        changed = upstream(signature)
        self.no_effect |= not changed or bool(self.exam.UNPLAYABLE.get('hit'))
        return changed

    def tap(self, x, y):
        point = int(x), int(y)
        skip_point = point == tuple(self.exam.SKIP_BUTTON.center)
        forced_skip = not self.executing and skip_point
        skip = None
        if forced_skip:
            image = self.host.image
            turns = self.box.request.state.turns_remaining
            if self.box.pending_action is not None:
                if not self.observe(image, allow_miss=True) or self.action_applied:
                    return
            screen = self.recognize(image).screen
            if (self.settlement is not None or screen is None or screen.observation.turns != turns
                    or self.refresh_live(screen, idle=True) is None):
                return
            request = self.box.request
            legal = self.legal(request)
            skip = next((a for a in legal if a.kind is ActionKind.END_TURN), None)
            if skip is None:
                return
        if skip_point:
            target = self.host.current_skip(self.host.image, self.box.request.state.turns_remaining)
            if target is None:
                return
            point = tuple(self.rect(target).center)
        elif self.drink_point is not None:
            if point == self.drink_point:
                # Hiro reuses this coordinate after waiting for calm, including
                # the one retap. Use the actual bottle on that latest screen.
                action = self.box.pending_action
                target = self.host.current_drink_target(self.host.image, action, self.box.request.state.turns_remaining)
                if target is None:
                    return
                point = target
            elif not self.action_taps:
                return
        if skip is not None:
            self.box.bind_host_action(request, skip, legal)
        self.last_job = self.host.click(point, purpose='hiro-exam' if not forced_skip else 'hiro-forced-skip')
        if self.executing or forced_skip:
            self.action_taps += 1
        if forced_skip:
            self.waiting_since = time.monotonic()
            self.feedback = self.host.action_feedback(self.box.request, self.box.pending_action, hiro_ui=True)

    def observe(self, image, *, allow_miss=False):
        request, action = self.box.request, self.box.pending_action
        if action is None:
            return True
        self.action_applied = False
        fresh = self.recognize(image).screen
        observed = None
        # Upstream can report instant change on a transient unreadable frame.
        # In that case reread for its original effect wait, then reconcile the
        # visible position only if it still shows that the action had no effect.
        waited = self.waiting_since is not None and (time.monotonic() - self.waiting_since >= self.exam.CHANGE_TIMEOUT)
        if self.action_taps:
            if self.feedback is not None:
                observed = self.feedback.read(image)
        elif fresh is not None:
            observed = self.host.reobserve_pending(image, request, action)
        if isinstance(observed, ExamResult):
            if observed.terminal_step is None:
                self.box.finish_on_observed_result(request.state, action, observed.score)
            else:
                self.box.observe_transition(request.state, action, observed.terminal_step)
                self.box.record_final_score(observed.score)
            self.settlement = ExamSettlement(observed.rank, self.box.final_score, self.box.request.state.score)
        elif isinstance(observed, ExamReobserved):
            self.box.reobserve_unsubmitted(request, action, observed.request)
        elif observed is not None:
            self.box.observe_transition(request.state, action, observed)
            self.action_applied = True
        elif fresh and (not self.action_taps or (allow_miss and (self.no_effect or waited))):
            if self.host.same_live(fresh, self.host.live_screen):
                self.box.discard_unsubmitted(request, action)
                self.host.live_screen = fresh
            elif not self.action_taps or (observed := self.host.reobserve_no_effect(request, action, fresh)) is None:
                return False
            else:
                self.box.reobserve_unsubmitted(request, action, observed.request)
        else:
            return False
        self.feedback, self.action_taps = None, 0
        self.no_effect = False
        self.waiting_since = None
        self.host.card_confirmation = None
        return True

    def ended(self, image):
        if self.settlement is not None:
            return True
        if not self.host.result_page_ready(image):
            return False
        if self.box.pending_action is not None:
            return self.observe(image) and self.settlement is not None
        # Some lessons and auditions transition after their last action has
        # already been reconciled. Read the displayed result without a retry.
        result = self.host.read_result_page(image)
        if result is None:
            return False
        state = self.box.request.state
        if state.terminal:
            self.box.record_final_score(result['score'])
        self.settlement = ExamSettlement(result['rank'], result['score'], state.score)
        return True

    # Hiro ExamStateSource and ExamAdvisor contracts.
    def begin(self, context):
        pass

    def read(self, frame, memory):
        idle = self.box.pending_action is None
        self.reconciling = frame.screen is not None and not self.observe(frame.img, allow_miss=True)
        if frame.screen is None or self.reconciling or self.settlement is not None:
            return None, 'maa_observation_pending'
        request = self.box.request
        if request.decision_kind is not DecisionKind.LIVE:
            return None, 'maa_choice_pending'
        request = self.refresh_live(frame.screen, idle=idle)
        if request is None:
            return None, 'maa_idle_position_not_aligned'
        memory.observe_turn(frame.remaining)
        # Use the physical bottle slot as part of the identity Hiro maps.
        drinks = [SimpleNamespace(slot=slot, drink_id=f'{slot}:{cid}', name=cid) for slot, cid in request.drink_slots]
        candidates = {key: action for action in self.legal(request)
                      if (key := _key(action)) is not None and key not in memory.blocked}
        if not candidates:
            return None, 'maa_no_legal_visible_action'
        return SimpleNamespace(hand=[SimpleNamespace(card_id=None, upgrade=None) for _ in frame.cards],
                               drinks=drinks, legal_actions=lambda: tuple(candidates), candidates=candidates), 'screen'

    def refresh_live(self, screen, *, idle):
        if self.host.same_live(screen, self.host.live_screen):
            self.host.live_screen = screen
        elif idle:
            # A game-driven change between actions must reach the policy before the next input.
            current = self.host.reobserve_visible(self.box.request, screen)
            if current is None:
                return None
            self.box.reobserve_idle(current)
            self.host.accept_live_screen(screen, current)
        return self.box.request

    def confidence(self):
        return 'screen'

    def animate(self, upstream, image):
        if self.reconciling and self.recognize(image).screen is not None:
            return False
        return upstream(image)

    def previews(self, state):
        return None

    def executed(self, action, result):
        if self.action_taps:
            self.host.log(dict(event='hiro_maa_action', action=action.legal_key(),
                               job_id=self.last_job, observed_change=result.ok))

    def end(self, result):
        pass

    def on_round_start(self, number, context):
        pass

    def decide(self, state, memory, previews, legal):
        selected = self.box.choose(self.box.request, tuple(state.candidates.values()))
        contract = next(action for action in legal if action.legal_key() == _key(selected))
        return self.exam.DecisionContract(contract, source='bench')

    def on_round_end(self, result):
        pass


class MaaExamFrame:
    def __init__(self, image, screen, cards, drinks):
        self.img, self.screen, self.cards, self.drinks = image, screen, cards, drinks
        self.t = time.monotonic()
        hud = screen.observation if screen is not None else None
        self.remaining, self.hp, self.genki = (hud.turns, hud.stamina, hud.genki) if hud else (None, None, None)
        self.score = hud.score if hud else None

    @property
    def signature(self):
        return self.remaining, self.hp, self.genki, tuple((c.card.name, c.available) for c in self.cards), tuple(
            drink.id for drink, _ in self.drinks), self.score


def run_maa_exam(host, box, legal, *, label):
    exam = hiro_module('exec.exam')
    memory = hiro_module('brain.state').ExamMemory
    bridge = MaaExam(exam, host, box, legal)
    binding = SimpleNamespace(screen=SimpleNamespace(shot=bridge.screenshot), device=bridge,
                              checkpoint=bridge.checkpoint, clock=time.monotonic, sleep=time.sleep)
    brain = SimpleNamespace(profile=None, new_memory=lambda name: memory(exam=name), commit=lambda *_: None)
    recorder = SimpleNamespace(log=lambda event, **data: host.log(dict(event=event, **{
        key: value for key, value in data.items() if key != 'image'})))
    strategy = hiro_module('strategy')
    # The Maa task selects the advisor and owns live observation and receipts.
    with (hiro_binding(binding), hiro_maa_ocr(lambda image: host.ocr(image)),
          _perception(strategy, _current=strategy.Strategy())):
        execute, persistent_select, wait_change = exam._execute, exam._select_retap, exam._wait_change
        animation = exam._anim_tap
        with _perception(exam, Frame=bridge.recognize, _lite_sig=bridge.signature, _roi_texts=bridge.texts,
                         _dialog_top=bridge.dialog_top, _drink_detail=bridge.drink_detail,
                         _handle_dialogs=bridge.dialogs, _usable=lambda frame: frame.screen is not None,
                         _execute=lambda *args: bridge.execute(execute, *args),
                         _select_retap=lambda marker: bridge.persistent_select(persistent_select, marker),
                         _wait_change=lambda signature: bridge.wait_change(wait_change, signature),
                         _anim_tap=lambda image: bridge.animate(animation, image),
                         SIG_FIELDS=(*exam.SIG_FIELDS, 'score'),
                         _RETAP={'pos': None, 'n': 0, 't0': 0.}, _STRIP={'img': None, 'sel': []}):
            exam.run_exam(brain, label, recorder, bridge.ended, source=bridge, advisor=bridge)
    if bridge.settlement is None:
        raise ExamNoProgress('hiro_exam_ended_without_game_result')
    return bridge.settlement
