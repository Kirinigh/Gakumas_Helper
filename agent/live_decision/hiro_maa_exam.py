"""Bridge Hiro's exam driver to Maa perception, dispatch and the strategy factory."""

import time
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import ExitStack, contextmanager
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


@contextmanager
def maa_tower_ranking():
    """Keep Hiro's Tower geometry and use the DMM player outline as a fallback."""
    from .idol_road_battle import read_idol_road_live_ranking

    vision = hiro_module('vision.tower')
    original = vision.my_rank

    def player(image):
        rank, confidence = original(image)
        if rank is None and (visible := read_idol_road_live_ranking(image, lambda *_: None)) is not None:
            rank = visible['rank']
        return rank, confidence

    with _perception(vision, my_rank=player):
        yield vision


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
        self.stale_streak = 0
        self.card_dispatch = None
        self.card_rejected = False

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
            read_strip = getattr(self.host, 'read_hand_strip', None)
            if read_strip is None:
                marks, _ = read_exam_select_markers(self.host.context, image)
            else:
                marks, _ = read_exam_select_markers(self.host.context, image, read_strip=read_strip)
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
        handled = self.exam._exam_dialogs(image, slow=slow, select=select and not self.card_rejected)
        action = self.box.pending_action
        if not handled and self.action_taps and action is not None:
            target = self.host.pending_dialog(image, action, slow=slow)
            if target is not None:
                self.tap(*self.rect(target).center)
                return True
        return handled

    def check_dispatch(self, upstream, frame, stale_limit):
        self.card_dispatch = None
        if self.stale_streak < stale_limit and (reason := upstream(frame)) is not None:
            self.stale_streak += 1
            return reason
        self.stale_streak = 0
        action = self.box.pending_action
        if action.kind is not ActionKind.PLAY_CARD:
            return None
        self.last_job = None
        self.exam.PLAY_RESULT['confirmed'] = False
        image = self.screenshot().img
        target = self.host.current_card_target(image, self.host.live_screen, action.source_slot)
        if target is None:
            self.card_rejected = True
            return '点牌前无法确认目标牌'
        fresh, slot, _ = target
        self.card_dispatch = (frame, action, self.recognized_frame(image, fresh), slot)
        return None

    def execute(self, upstream, decision, frame):
        action = self.box.pending_action
        self.last_job = None
        if action.kind is ActionKind.PLAY_CARD:
            checked, self.card_dispatch = self.card_dispatch, None
            if checked is None or checked[0] is not frame or checked[1] != action:
                raise ExamNoProgress('maa_exam_card_dispatch_unverified')
            _, _, frame, slot = checked
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
        if action is not None and self.host.card_confirmation is not None:
            # The current marker can belong to a different card after the confirmed play.
            return False
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
        unconfirmed = (self.action_taps and action.kind is ActionKind.PLAY_CARD
                       and self.host.card_confirmation is None)
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
        elif fresh:
            # Hiro returns to its exam loop after a first tap without SELECT.
            # The fresh position can clear this intent without another effect wait.
            if self.action_taps and not allow_miss:
                return False
            if self.action_taps and not (unconfirmed or self.no_effect) and (
                    self.waiting_since is None
                    or time.monotonic() - self.waiting_since < self.exam.CHANGE_TIMEOUT):
                return False
            if self.host.same_live(fresh, self.host.live_screen):
                self.box.discard_unsubmitted(request, action)
                self.host.live_screen = fresh
            else:
                if not self.action_taps or (observed := self.host.reobserve_no_effect(
                        request, action, fresh, unconfirmed_image=image if unconfirmed else None)) is None:
                    return False
                self.box.reobserve_unsubmitted(request, action, observed.request)
        else:
            return False
        self.feedback, self.action_taps = None, 0
        self.no_effect = False
        self.waiting_since = None
        self.card_rejected = False
        self.host.card_confirmation = None
        return True

    def ended(self, image):
        if self.settlement is not None:
            return True
        hiro_lesson_ended = getattr(self.host, 'hiro_lesson_ended', None)
        lesson_finished = hiro_lesson_ended(image) if hiro_lesson_ended is not None else False
        if not lesson_finished and not self.host.result_page_ready(image):
            return False
        pending = self.box.pending_action is not None
        if pending:
            if self.observe(image) and self.settlement is not None:
                return True
            read = getattr(self.host, 'read_unscored_failure', None)
            result = read(image) if read is not None else None
        else:
            result = self.host.read_result_page(image)
        # Some lessons and auditions transition after their last action has
        # already been reconciled. Read the displayed result without a retry.
        if result is None:
            if lesson_finished:
                # Hiro confirmed that the lesson scene ended. An unread
                # gain does not settle a pending action or fabricate a score.
                self.settlement = ExamSettlement(None, None, self.box.request.state.score)
                return True
            return False
        state = self.box.request.state
        if result.get('failed') and result['score'] is None:
            self.settlement = ExamSettlement(result['rank'], None, state.score, failed=True)
            return True
        if pending:
            return False
        if result['score'] is not None:
            self.box.record_final_score(result['score'], observed_result=True)
        self.settlement = ExamSettlement(result['rank'], result['score'], state.score)
        return True

    # Hiro ExamStateSource and ExamAdvisor contracts.
    def begin(self, context):
        self.session = context['session']
        self.stale_streak = 0

    def read(self, frame, memory):
        idle = self.box.pending_action is None
        self.reconciling = frame.screen is not None and not self.observe(frame.img, allow_miss=True)
        if frame.screen is None or self.reconciling or self.settlement is not None:
            return None, 'maa_observation_pending'
        request = self.box.request
        if request.decision_kind is not DecisionKind.LIVE:
            return None, 'maa_choice_pending'
        request = self.refresh_live(frame, idle=idle)
        if request is None:
            return None, 'maa_idle_position_not_aligned'
        memory.observe_turn(frame.remaining)
        # Use the physical bottle slot as part of the identity Hiro maps.
        native = getattr(self.host, 'native_rules', None)
        drink_ids = native.drink_forms if native is not None else {}
        drinks = [SimpleNamespace(slot=slot, drink_id=drink_ids.get(cid), name=cid, binding=f'{slot}:{cid}')
                  for slot, cid in request.drink_slots]
        candidates = {key: action for action in self.legal(request)
                      if (key := _key(action)) is not None and key not in memory.blocked}
        if not candidates:
            return None, 'maa_no_legal_visible_action'
        save_position = getattr(self.host, 'save_committed_exam', None)
        if save_position is not None:
            save_position(request, frame.screen.observation)
        hud = frame.screen.observation
        if getattr(self, 'session', None) is not None:
            self.session.observe(scene='HIF_EXAM', img=frame.img, ready=True, save_frame=False, fields={
                'remaining_turns': hud.turns, 'hp': hud.stamina, 'genki': hud.genki, 'score': hud.score,
                'hand': [{'slot': slot, 'card_id': cid, 'available': slot in hud.playable}
                         for slot, cid in enumerate(hud.hand)],
                'drinks': [{'slot': slot, 'drink_id': drink_ids.get(cid)} for slot, cid in hud.drinks],
                'buffs': dict(hud.statuses),
            })
        return SimpleNamespace(hand=self.card_forms(request, hud.hand),
                               drinks=drinks, legal_actions=lambda: tuple(candidates), candidates=candidates), 'screen'

    def card_forms(self, request, visible):
        """Supply Hiro's effect check with the master form bound to each physical slot."""
        from .idol_road_battle import _wire_card

        if not visible:
            return []
        cards = {card.instance_id: card for card in request.state.deck.hand}
        slots = {slot: cards[guid] for guid, slot in request.card_slots}
        session = getattr(self.host, 'native_session', None)
        native_forms = ({guid: (identifier, upgrade) for guid, identifier, upgrade, _ in session.native.cards_info()}
                        if session is not None else None)
        hand = []
        for slot, identifier in enumerate(visible):
            card = slots.get(slot)
            form = (None, None)
            if card is not None and card.card_id == identifier:
                if native_forms is not None:
                    form = native_forms.get(card.instance_id, form)
                else:
                    try:
                        form = _wire_card(self.host.kernel, card.card_id, card.upgraded)
                    except ValueError:
                        pass  # Hiro uses its conservative effect check for an unknown form.
            hand.append(SimpleNamespace(card_id=form[0], upgrade=form[1]))
        return hand

    def screen_slot(self, upstream, frame, state, decision):
        if decision.kind != 'drink':
            return upstream(frame, state, decision)
        target = next((drink.binding for drink in state.drinks if drink.slot == decision.slot), None)
        return next((index for index, (drink, _) in enumerate(frame.drinks) if drink.id == target), None)

    def refresh_live(self, frame, *, idle):
        screen = frame.screen
        if self.host.same_live(screen, self.host.live_screen):
            self.host.live_screen = screen
        elif idle:
            # A game-driven change between actions must reach the policy before the next input.
            observed = self.host.reobserve_visible(frame.img, self.box.request, screen)
            if observed is None:
                return None
            current, corrected = observed
            self.box.reobserve_idle(current)
            self.host.accept_live_screen(corrected, current)
        if self.host.live_screen != screen:
            # The same image can be reconciled using the known card forms.
            updated = self.recognized_frame(frame.img, self.host.live_screen)
            frame.screen, frame.cards, frame.drinks = updated.screen, updated.cards, updated.drinks
        return self.box.request

    def confidence(self):
        return 'screen'

    def animate(self, upstream, image):
        if self.card_rejected or (self.reconciling and self.recognize(image).screen is not None):
            return False
        return upstream(image)

    def previews(self, state):
        return None

    def executed(self, action, result):
        if self.action_taps:
            self.host.log(dict(event='hiro_maa_action', action=action.legal_key(),
                               job_id=self.last_job, observed_change=result.ok,
                               card_confirmation_sent=self.host.card_confirmation is not None))

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


@contextmanager
def maa_exam_session(log, session=None):
    if session is not None:
        yield session
        return
    plugins = hiro_module('plugin_record')
    with TemporaryDirectory(prefix='gkh-hiro-exam-') as directory:
        trace = SimpleNamespace(dir=Path(directory), seq=0)

        def next_seq():
            trace.seq += 1
            return trace.seq

        def write(step):
            if step.execution is not None:
                log(dict(event='hiro_exam_step', seq=step.seq, notes=step.notes,
                         timings_ms=step.timings_ms, execution=step.execution.to_dict()))

        trace.next_seq, trace.write, trace.save_frame = next_seq, write, lambda *_: None
        with _perception(plugins, RECORD=plugins.Record()):
            yield hiro_module('loop.session').Session(trace, clock=time.monotonic, sleep=time.sleep)


def _run_with_scene_reread(run, screenshot, ended, in_exam, settled, log):
    run()
    if settled():
        return
    image = screenshot()
    if ended(image) or not in_exam(image):
        return
    log(dict(event='hiro_exam_reentered', reason='current_exam_still_visible'))
    run()
    if not settled():
        ended(screenshot())


def game_recommendation(host, *, opening_drinks=False):
    """Select from Hiro's observed slots using the game's visual recommendation."""
    from .exam_hand import anchor_exam_hand

    started = time.monotonic()

    def choose(frame, state):
        nonlocal opening_drinks
        if opening_drinks:
            bottle = next((key for drink in state.drinks if (key := f'drink:{drink.slot}') not in state.blocked),
                          None)
            if bottle is not None:
                return bottle
            opening_drinks = False
        visible = [(slot, card) for slot, card in enumerate(frame.cards)
                   if card is not None and card.available and f'play:{slot}' not in state.blocked]
        if not visible:
            return 'skip' if 'skip' not in state.blocked else None
        match = host.context.run_recognition('ProduceRecognitionCards', frame.img)
        rows = getattr(match, 'all_results', ()) if getattr(match, 'hit', False) else ()
        rows = anchor_exam_hand(frame.img, rows, host) or rows
        ranked = []
        for row in rows:
            if row.label not in ('suggestions', 'cards'):
                continue
            x, _, w, _ = row.box
            center = x + w / 2
            slot, card = min(visible, key=lambda pair: abs(pair[1].rect.center.x - center))
            if abs(card.rect.center.x - center) <= min(w, card.rect.x2 - card.rect.x1) / 2:
                ranked.append((row.label == 'suggestions', row.score, slot))
        if ranked:
            recommended, _, slot = max(ranked)
            if recommended or len(visible) == 1 or time.monotonic() - started >= 15.:
                return f'play:{slot}'
        if len(visible) == 1 or time.monotonic() - started >= 15.:
            return f'play:{visible[0][0]}'
        return None

    return choose


def run_maa_exam(host, box, legal, *, label, session=None):
    exam = hiro_module('exec.exam')
    memory = hiro_module('brain.state').ExamMemory
    bridge = MaaExam(exam, host, box, legal)
    binding = SimpleNamespace(screen=SimpleNamespace(shot=bridge.screenshot), device=bridge,
                              checkpoint=bridge.checkpoint, clock=time.monotonic, sleep=time.sleep)
    brain = SimpleNamespace(profile=None, new_memory=lambda name: memory(exam=name), commit=lambda *_: None)
    recorder = SimpleNamespace(log=lambda event, **data: host.log(dict(event=event, **{
        key: value for key, value in data.items() if key != 'image'})))
    strategy = hiro_module('strategy')
    config = strategy.Strategy()
    with (hiro_binding(binding), hiro_maa_ocr(host.ocr), maa_exam_session(host.log, session) as session,
          _perception(strategy, current=lambda: config)):
        execute, persistent_select, wait_change = exam._execute, exam._select_retap, exam._wait_change
        stale_before_tap, stale_limit = exam._stale_before_tap, exam.STALE_MAX
        screen_slot = exam._screen_slot
        animation = exam._anim_tap
        with _perception(exam, Frame=bridge.recognize, _lite_sig=bridge.signature, _roi_texts=bridge.texts,
                         _dialog_top=bridge.dialog_top, _drink_detail=bridge.drink_detail,
                         _handle_dialogs=bridge.dialogs, _usable=lambda frame: frame.screen is not None,
                         # Always run Maa's last check. The bridge keeps Hiro's own stale retry limit.
                         STALE_MAX=float('inf'),
                         _stale_before_tap=lambda frame: bridge.check_dispatch(stale_before_tap, frame, stale_limit),
                         _execute=lambda *args: bridge.execute(execute, *args),
                         _screen_slot=lambda *args: bridge.screen_slot(screen_slot, *args),
                         _select_retap=lambda marker: bridge.persistent_select(persistent_select, marker),
                         _wait_change=lambda signature: bridge.wait_change(wait_change, signature),
                         _anim_tap=lambda image: bridge.animate(animation, image),
                         SIG_FIELDS=(*exam.SIG_FIELDS, 'score'),
                         _RETAP={'pos': None, 'n': 0, 't0': 0.}, _STRIP={'img': None, 'sel': []}):
            def run():
                exam.run_exam(brain, label, recorder, bridge.ended, session=session, source=bridge, advisor=bridge)

            _run_with_scene_reread(run, lambda: bridge.screenshot().img, bridge.ended,
                                   lambda image: bridge.recognize(image).screen is not None,
                                   lambda: bridge.settlement is not None, host.log)
    if bridge.settlement is None:
        raise ExamNoProgress('hiro_exam_ended_without_game_result')
    return bridge.settlement


def run_maa_screen_exam(host, read_result, *, label, retry=False, session=None, capture=None, classify=None,
                        policy='hiro-screen', opening_drinks=False):
    """Hiro's screen-only state source and offline rules on the existing Maa task."""
    from .hiro_maa_device import HiroMaaDevice, hiro_maa_route
    from .high_score.policy_factory import build_battle_policy

    exam = hiro_module('exec.exam')
    screen_exam = hiro_module('defaults.exam_state')
    base_source = screen_exam.ScreenExamStateSource
    brain, advisor, recommend = build_battle_policy(policy, backend='screen', host=host,
                                                   opening_drinks=opening_drinks)

    class ScreenSource(base_source):
        def read(self, frame, memory):
            raw = getattr(frame, 'raw', None) or screen_exam.read_screen(frame.img)
            frame.raw = host.fill_hiro_hand_slots(frame.img, raw, frame.cards)
            if frame.raw is not raw:
                frame._hand = frame.raw['hand']
            state, cause = super().read(frame, memory)
            if state is None or recommend is None:
                return state, cause
            choice = recommend(frame, state)
            if choice is None:
                return None, 'game_recommendation_unread'
            state.legal_actions = lambda: (choice,)
            return state, cause

    def authorize():
        if not host._authorized():
            raise ValueError('the active Maa input endpoint changed')

    result = None

    def ended(image):
        nonlocal result
        result = read_result(image)
        return result is not None

    device = HiroMaaDevice(host.context if capture is not None else host.base, authorize, log=host.log)
    if capture is not None:
        device._capture = capture
    shot = device.screenshot

    def screenshot():
        image = shot()
        host.check_tower_date(image)
        return image

    device.screenshot = screenshot
    recorder = SimpleNamespace(log=lambda event, **data: host.log(dict(event=event, **{
        key: value for key, value in data.items() if key != 'image'})))
    with (hiro_maa_route(device, host.ocr), maa_tower_ranking(), maa_exam_session(host.log, session) as session,
          ExitStack() as providers):
        if hasattr(exam, 'pick_advisor'):
            strategy = hiro_module('strategy')
            config = strategy.Strategy()
            providers.enter_context(_perception(strategy, current=lambda: config))
        if retry:
            host.enter_same_formation_retry()
        source = ScreenSource()

        def run():
            exam.run_exam(brain, label, recorder, ended, session=session, source=source, advisor=advisor)

        def in_exam(image):
            hit = (classify or hiro_module('runtime.tower_play').tower_classify)(image)
            return hit is not None and hit.name in ('EXAM', 'HIF_EXAM', 'TOWER_EXAM')

        _run_with_scene_reread(run, screenshot, ended, in_exam, lambda: result is not None, host.log)
    if result is None:
        raise ExamNoProgress('hiro_exam_ended_without_game_result')
    return result
