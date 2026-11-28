"""Run the pinned Hiro page dispatcher inside the current Maa Produce task."""

import time
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import ExitStack, contextmanager
from dataclasses import replace

from agent.live_decision.hiro_runtime import HIRO_REVISION, hiro_module
from agent.live_decision.hiro_maa_device import hiro_maa_route


def _capture_sink(log):
    import cv2
    from utils.local_data import LocalDataError, get_runtime_store, get_runtime_config

    mode = get_runtime_config().mode
    if mode == 'off':
        return None
    store = get_runtime_store()
    session = None
    disabled = False

    def save(kind, seq, image):
        nonlocal session, disabled
        if disabled or image is None or (kind == 'screenshot' and mode != 'screenshot'):
            return None
        try:
            ok, data = cv2.imencode('.jpg', image)
            if ok:
                if session is None:
                    session = store.start_session('hiro-maa-cultivation')
                return str(store.write_image('failure' if kind == 'failure_scene' else 'normal',
                    session, data.tobytes(), capture_kind=kind, suffix='.jpg', metadata={'sequence': seq}))
        except (LocalDataError, OSError, ValueError, cv2.error) as error:
            disabled = True
            log(dict(event='hiro_capture_unavailable', reason=str(error)))
        return None

    return save


def nia_fallback(context, authorize):
    """Use the local Maa handlers for NIA pages uncovered by the pinned router."""
    def advance(rt, frame, hit):
        if hit.name != 'unknown':
            return False
        for page, node in (('ProduceMirrorFlag', 'ProduceMirrorFlag'), ('ProduceGuideFlag', 'ProduceGuideExit')):
            recognized = context.run_recognition(page, frame.img)
            if not recognized or not recognized.hit:
                continue
            target = recognized if node == page else context.run_recognition(node, frame.img)
            if not target or not target.hit:
                return False
            authorize()
            action = context.run_action(node, box=target.best_result.box)
            if not action or action.success is not True:
                raise hiro_module('devices.base').DeviceError('Maa NIA fallback outcome is unknown')
            return True
        return False

    return advance


def maa_next_nodes(context, name):
    """Maa expands each configured next target into a node descriptor."""
    targets = (context.get_node_data(name) or {}).get('next', ())
    return tuple(target['name'] if isinstance(target, dict) else target for target in targets)


def selected_reward_policy(context):
    settings = (context.get_node_data('ProduceHiroRewardSettings') or {}).get('attach') or {}
    policy = settings.get('policy', 'hiro')
    if policy not in ('hiro', 'recommended'):
        raise ValueError('unknown cultivation reward policy')
    return policy


@contextmanager
def _replace(module, name, value):
    missing = object()
    original = getattr(module, name, missing)
    setattr(module, name, value)
    try:
        yield
    finally:
        if original is missing:
            delattr(module, name)
        else:
            setattr(module, name, original)


@contextmanager
def _hiro_diagnostics(log):
    """Route the pinned runtime's Python logs through the same diagnostic sink."""
    upstream = logging.getLogger('hiro')

    class Handler(logging.Handler):
        def emit(self, record):
            log(dict(event='hiro_runtime_diagnostic', source=record.name,
                     level=record.levelname, message=record.getMessage()))

    handler = Handler()
    propagate = upstream.propagate
    upstream.addHandler(handler)
    upstream.propagate = False
    try:
        yield
    finally:
        upstream.removeHandler(handler)
        upstream.propagate = propagate


@contextmanager
def _hiro_captures(trace):
    """Keep Hiro's calibration paths valid and send their images to Maa's capture sink."""
    ops = hiro_module('produce.ops')
    imgio = hiro_module('imgio')
    review = hiro_module('produce.deck_review')
    card_ops = hiro_module('produce.card_ops')
    directory = trace.dir
    original_write = imgio.imwrite

    def write_image(path, image, params=None):
        if not Path(path).is_relative_to(directory):
            return original_write(path, image, params)
        return trace.save_frame(trace.next_seq(), image) is not None

    def snap(run, image, _tag):
        if run.calib is not None:
            trace.save_frame(trace.next_seq(), image)

    with ExitStack() as scope:
        scope.enter_context(_replace(ops, 'CALIB_DIR', directory / 'calib'))
        scope.enter_context(_replace(ops.Run, 'snap', snap))
        scope.enter_context(_replace(imgio, 'imwrite', write_image))
        scope.enter_context(_replace(review, 'SAVE_DIR', None))
        # Hiro launches each segment in a new process. Keep these native
        # counters with the segment when Maa runs successive jobs in-process.
        scope.enter_context(_replace(card_ops._change_run, 'run', None))
        scope.enter_context(_replace(card_ops._change_run, 't', 0.))
        scope.enter_context(_replace(card_ops, '_STAGE2_CANCELS', {}))
        scope.enter_context(_replace(card_ops, 'LAST_CHANGED', None))
        yield


def _card_change_index(deck, args):
    old, new, upgrade = args.get('remove_card_id'), args.get('card_id'), args.get('remove_upgrade')
    if deck is not None and old and new and type(upgrade) is int:
        matches = [i for i, card in enumerate(deck)
                   if card.get('card_id') == old and card.get('upgrade') == upgrade]
        if matches and all(deck[i] == deck[matches[0]] for i in matches):
            return matches[0]
    return None


def _mark_deck_pending(state):
    # Hiro already reviews pending decks on a subsequent action page.
    if 'produce.deck' not in state.pending_review:
        state.pending_review.append('produce.deck')
    state.field_status['produce.deck'] = 'pending'


@contextmanager
def _hiro_card_changes():
    """Adapt explicit selection conflicts and actual exchange receipts to the pinned Hiro flow."""
    card_ops = hiro_module('produce.card_ops')
    vision = hiro_module('vision.deck')
    ledger = hiro_module('loop.events')
    prediction = hiro_module('state.events')
    session = hiro_module('loop.session').Session
    event_class = hiro_module('contracts').Event
    original_change = card_ops.change_stage2
    original_events = ledger.events_from_step
    original_apply = ledger._apply_deck
    original_predict = prediction.PREDICTORS['CARD_CHANGE']
    original_record = session.record
    logger = logging.getLogger(card_ops.__name__)

    class SelectionConflict(Exception):
        pass

    def change(*args, **kwargs):
        original_read = vision.read_card_dialog
        selected = None

        def read(*args, **kwargs):
            nonlocal selected
            result = original_read(*args, **kwargs)
            if kwargs.get('detail_box') == vision.CHANGE2_DETAIL_BOX:
                if result.get('agree') is False:
                    raise SelectionConflict
                selected = result.get('selected_card')
            return result

        try:
            with _replace(vision, 'read_card_dialog', read):
                changed = original_change(*args, **kwargs)
        except SelectionConflict:
            logger.warning('换卡阶段 2：卡格与详情栏冲突，取消这次选择')
            card_ops.device.click(*card_ops.CHANGE_CANCEL_TAP)
            return False
        actual = card_ops.LAST_CHANGED
        if changed and actual and selected and selected.get('card_id') == actual['card_id']:
            if type(selected.get('upgrade')) is int:
                actual['upgrade'] = selected['upgrade']
        return changed

    def events(page, action, ok, page_fields, produce, *, seq):
        observed = original_events(page, action, ok, page_fields, produce, seq=seq)
        for index, event in enumerate(observed):
            if event.kind != 'CARD_CHANGE':
                continue
            facts = dict(event.args)
            actual = card_ops.LAST_CHANGED or {}
            upgrade = actual.get('upgrade') if actual.get('card_id') == facts.get('remove_card_id') else None
            facts.pop('remove_upgrade', None)
            facts.pop('upgrade', None)
            if type(upgrade) is int:
                facts.update(remove_upgrade=upgrade, upgrade=upgrade)
            observed[index] = replace(event, args=facts)
        return observed

    def apply(state, event):
        if event.kind != 'CARD_CHANGE':
            return original_apply(state, event)
        deck = (state.produce or {}).get('deck')
        args = event.args
        index = _card_change_index(deck, args)
        if index is None:
            _mark_deck_pending(state)
        else:
            deck.pop(index)
            deck.append(dict(card_id=args['card_id'], upgrade=args['remove_upgrade']))

    def predict(produce, args, seq):
        if _card_change_index(produce.get('deck'), args) is None:
            return []
        return original_predict(produce, args, seq)

    def record(self, obs=None, *, execution=None, **kwargs):
        if execution is not None and execution.ok and execution.action.args.get('confirm_late'):
            ledger._CHANGE_IN.clear()
            _mark_deck_pending(self.state)
            self.emit(event_class(seq=(obs or self.last).seq, kind='UNMODELED_CHANGE', origin='action',
                                  args={'field': 'produce.deck'}, note='换卡已补发延迟确认，实际生效和实体待核'))
        return original_record(self, obs, execution=execution, **kwargs)

    with (
        _replace(ledger, '_CHANGE_IN', {}),
        _replace(card_ops, 'change_stage2', change),
        _replace(ledger, 'events_from_step', events),
        _replace(ledger, '_apply_deck', apply),
        _replace(prediction, 'PREDICTORS', {**prediction.PREDICTORS, 'CARD_CHANGE': predict}),
        _replace(session, 'record', record),
    ):
        yield


@contextmanager
def _hiro_memory_indexes():
    """Keep Hiro's two asset indexes in the client's versioned local runtime."""
    from agent.utils.local_data import DEFAULT_RUNTIME_ROOT

    memory = hiro_module('vision.memory_icons')
    renderer = hiro_module('vision.card_icon_render')
    directory = DEFAULT_RUNTIME_ROOT / 'hiro-formation' / HIRO_REVISION
    with (
        _replace(memory, 'ART_CACHE', str(directory / 'memory-icons.npz')),
        _replace(renderer, 'CACHE', directory / 'card-icons.npz'),
    ):
        memory._art_index.cache_clear()
        memory._render_index.cache_clear()
        try:
            yield
        finally:
            memory._art_index.cache_clear()
            memory._render_index.cache_clear()


class _Trace:
    """Keep upstream state and events without creating its screenshot archive."""

    def __init__(self, directory, mode, log, save_frame):
        self.dir, self.log, self.save = Path(directory), log, save_frame
        self.mode = mode
        self.seq = 0
        self.r1_recorded = False
        self.exam_retries = {}

    def next_seq(self):
        self.seq += 1
        return self.seq

    def save_frame(self, seq, image):
        return self.save('screenshot', seq, image) if self.save else None

    def write(self, step):
        page = step.observation.page.name
        fields = step.observation.fields or {}
        if page == 'HIF_FINALS_RANKING' and fields.get('kind') == 'round_result':
            self.r1_recorded = True
        if step.events or step.decision is not None or step.execution is not None:
            events = []
            for observed in step.events:
                event = observed.to_dict()
                if observed.args.get('memory_effect'):
                    applied = [item for item in (step.state_after or step.state).ledger
                               if item.seq == observed.seq and item.kind == 'CARD_UPGRADE'
                               and item.args.get('source') == 'memory'
                               and item.args.get('card_id') == observed.args.get('card_id')]
                    if len(applied) == 1:
                        event['memory_upgrade_from'] = applied[0].args.get('upgrade')
                events.append(event)
            self.log(dict(event='hiro_cultivation_events', segment=self.mode, page=page, sequence=step.seq,
                          decision=step.decision.to_dict() if step.decision is not None else None,
                          execution=step.execution.to_dict() if step.execution is not None else None,
                          events=events))


def _retry_unscored_failure(core, label, mode):
    """Use the current Hiro result page and its original retry rules."""
    image = hiro_module('devices.active').device.screenshot()
    texts = hiro_module('vision.ocr').screen_texts(image)
    if not any('不合格' in ''.join(t.text.split()) and 20 <= t.rect.center.x <= 260
               and 120 <= t.rect.center.y <= 200 for t in texts):
        return False
    button_text = hiro_module('produce.ops').button_text
    button = next((t for t in texts if button_text(t.text, '再挑戦') and t.rect.center.y > 900), None)
    if button is None or (kind := core._click_retry(label, button, texts)) is None:
        return False
    took_effect = getattr(core, '_retry_took_effect', None)
    if mode == 'nia' and took_effect is not None and not took_effect(label):
        return False
    core.exam_restarted = True
    core.exam_count -= 1
    type(core).exams_done = core.exam_count
    core.rec.log('exam_retry', exam=label, retry_kind=kind)
    return True


def _new_runtime(rt, trace, *, mode, difficulty, skin, main_attr, play_exam, fallback, log, retry, reward_policy):
    play = hiro_module('runtime.play')
    nia = hiro_module('runtime.nia_sample')
    hif = hiro_module('produce.hif')
    session = hiro_module('loop.session').Session(trace, clock=rt.clock, sleep=rt.sleep)
    rt.session = session

    unscored_failure = None

    def battle(_brain, label, _recorder, ended, *, session, advisor):
        nonlocal unscored_failure
        unscored_failure = None
        result = play_exam(rt, label, session, ended)
        if getattr(result, 'failed', False) and result.score is None:
            unscored_failure = label
        return result

    if mode == 'hajime':
        hajime = hiro_module('runtime.hajime')
        # The learning-page handler captures the upstream function when built.
        with _replace(hiro_module('exec.exam'), 'run_exam', battle):
            parent = hajime.hajime_core(hajime.MODES[difficulty])
        nia.set_mode(session, hajime.MODES[difficulty])
    elif mode == 'nia':
        parent = nia.nia_core(nia.MODES[difficulty])
        nia.set_mode(session, nia.MODES[difficulty])
    else:
        parent = hif.HifCore

    class Core(parent):
        def _exam_advisor(self, llm):
            # The shared Maa battle adapter supplies the GKH policy.
            return None

        def _r1_done_in_trace(self):
            return trace.r1_recorded

        def _retry_if_failed(self, label, **kwargs):
            nonlocal unscored_failure
            if mode in ('hajime', 'nia') and label == unscored_failure:
                unscored_failure = None
                return retry and _retry_unscored_failure(self, label, mode)
            return retry and super()._retry_if_failed(label, **kwargs)

        def _retries_so_far(self, label):
            # Hiro's CLI counts previous Step notes on disk. The Maa trace
            # avoids those files and retains the confirmed retry event instead.
            return trace.exam_retries.get(label, 0)

    class Recorder:
        dir = trace.dir

        @staticmethod
        def log(event, *, image=None, **data):
            if event == 'finals_round_result' and data.get('exam') == 'r1' and data.get('page_kind') == 'round_result':
                trace.r1_recorded = True
            if event == 'exam_retry':
                label = data['exam']
                trace.exam_retries[label] = trace.exam_retries.get(label, 0) + 1
            log(dict(event=event, **data))

    Core.session = session
    Core.recorder = Recorder()
    Core.use_llm = False
    Core.exams_done = 0
    Core.lessons_done = 0
    Core.main_attr = main_attr
    core = Core(scenario='main' if mode == 'main' else 'qualify', idol=skin)
    if reward_policy == 'recommended':
        original = core.deciders.select_card
        contracts = hiro_module('contracts')

        def select_card(state, *, change=False):
            choices = state.page_fields.get('candidates') or ()
            target = state.page_fields.get('recommended')
            if not change and type(target) is int and 0 <= target < len(choices):
                return contracts.Decision(contracts.Action.activate_effect(target), '游戏推荐奖励卡',
                                          'recommended', 1.0, decider='select_card')
            return original(state, change=change)

        core.deciders.select_card = select_card
    rt.handlers = hiro_module('produce.handlers').make_handlers(core)
    rt.advance = play.advance
    rt.idle_stop_s = play.IDLE_STOP_S
    rt.landscape = hiro_module('loop.landscape').LandscapeGuard(active=False)
    if mode in ('hajime', 'nia'):
        class Classifier(nia.Sampler):
            def __init__(self, classify):
                self.classify, self.nia = classify, mode == 'nia'
                self.out = trace.dir

            def save(self, img, hit, force=False):
                if trace.save:
                    trace.save_frame(trace.next_seq(), img)

        rt.classify = classifier = Classifier(rt.classify)
        if mode == 'nia':
            rt.handlers.update(hiro_module('produce.nia').NiaPages(core, persist=None).handlers())
            original = nia.make_advance(classifier)

            def advance(runtime, frame, hit):
                if (fallback and hit.name == 'unknown' and not nia._connecting(frame.img)
                        and fallback(runtime, frame, hit)):
                    return True
                return original(runtime, frame, hit)

            rt.advance = advance
    return battle


def run_hiro_cultivation(device, read_ocr, *, mode, skin, difficulty=None, main_attr=None, enter=None,
                         play_exam, log, fallback=None, save_frame=None, retry=True,
                         opening_submitted=None, reward_policy='hiro', resume_saved=False,
                         clock=time.monotonic, sleep=time.sleep):
    """Enter a mode and use its original handlers through settlement.

    The caller owns the confirmed opening and its AP receipt. After Hiro
    returns from title, only the current save is resumed.
    """
    if mode not in ('qualify', 'main', 'hajime', 'nia'):
        raise ValueError('unknown Hiro cultivation mode')
    if reward_policy not in ('hiro', 'recommended'):
        raise ValueError('unknown cultivation reward policy')
    play = hiro_module('runtime.play')
    finish = hiro_module('runtime.finish')
    date = hiro_module('loop.date_change')
    nia = hiro_module('runtime.nia_sample')
    with TemporaryDirectory(prefix='gkh-hiro-') as directory, ExitStack() as scope:
        rt = scope.enter_context(hiro_maa_route(device, read_ocr, clock=clock, sleep=sleep))
        scope.enter_context(_hiro_diagnostics(log))
        if mode in ('qualify', 'main'):
            from .hiro_card_rewards import hif_reward_options

            scope.enter_context(_replace(hiro_module('vision.card_select'), 'select_card_options',
                                         hif_reward_options(skin)))
        trace = _Trace(directory, mode, log, save_frame)
        finals = hiro_module('state.finals_context')
        scope.enter_context(_replace(finals, 'CONTEXT', {}))
        scope.enter_context(_hiro_captures(trace))
        scope.enter_context(_hiro_card_changes())
        scope.enter_context(_hiro_memory_indexes())
        scope.enter_context(_replace(hiro_module('produce.schedule_probe'), 'enabled', lambda: False))
        # The client's settings above choose the cultivation policy. Hiro's
        # standalone strategy file must not override it or activate a plugin.
        scope.enter_context(_replace(hiro_module('brain.plugin_gate'), 'produce_decide',
                                     lambda *args, **kwargs: (None, None)))
        scope.enter_context(_replace(hiro_module('runtime.retry_handoff'), 'enabled', lambda: False))
        scope.enter_context(_replace(nia, '_save_stop_frame', lambda rt, sampler, frame, hit:
            trace.save('failure_scene', trace.next_seq(), frame.img) if trace.save else None))
        battle = _new_runtime(rt, trace, mode=mode, difficulty=difficulty, skin=skin, main_attr=main_attr,
                              play_exam=play_exam, fallback=fallback, log=log, retry=retry, reward_policy=reward_policy)
        scope.enter_context(_replace(hiro_module('produce.hif'), 'run_exam', battle))
        started = enter is None
        for attempt in range(play.MAX_DATE_CHANGES + 1):
            try:
                if attempt:
                    play.recover_date_change(rt, rt.session)
                if not started:
                    enter(rt)
                    started = True
                elif attempt or enter is None:
                    finish.resume_produce(rt)
                    if resume_saved and not attempt:
                        scene = rt.classify(rt.screen.shot().img)
                        rt.gkh_resume_exam = scene.name in ('EXAM', 'HIF_EXAM', 'TOWER_EXAM', 'unknown')
                rt.run(until=lambda: getattr(rt, 'finished', False))
                return rt.screen.frame
            except date.DateChanged:
                # An interruption before the AP click may re-enter opening.
                # After a charged click, follow the same save.
                if opening_submitted is not None:
                    started = started or opening_submitted()
            except date.DataLinkBlocked as error:
                raise hiro_module('runtime.loop').ResumeBlocked(str(error)) from error
        raise hiro_module('runtime.loop').ResumeBlocked('Hiro cultivation date recovery exhausted')
