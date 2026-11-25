"""Run the pinned Hiro page dispatcher inside the current Maa Produce task."""

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import ExitStack, contextmanager

from agent.live_decision.hiro_runtime import hiro_module
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
        except (LocalDataError, OSError, ValueError) as error:
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
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


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


def _new_runtime(rt, trace, *, mode, difficulty, skin, main_attr, play_exam, fallback, log, retry, reward_policy):
    play = hiro_module('runtime.play')
    nia = hiro_module('runtime.nia_sample')
    hif = hiro_module('produce.hif')
    session = hiro_module('loop.session').Session(trace, clock=rt.clock, sleep=rt.sleep)
    rt.session = session

    def battle(_brain, label, _recorder, ended, *, session, advisor):
        return play_exam(rt, label, session, ended)

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
                         opening_submitted=None, reward_policy='hiro', clock=time.monotonic, sleep=time.sleep):
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
        trace = _Trace(directory, mode, log, save_frame)
        finals = hiro_module('state.finals_context')
        scope.enter_context(_replace(finals, 'CONTEXT', {}))
        scope.enter_context(_replace(hiro_module('produce.ops'), 'CALIB_DIR', None))
        scope.enter_context(_replace(hiro_module('produce.deck_review'), 'SAVE_DIR', None))
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
                rt.run(until=lambda: getattr(rt, 'finished', False))
                return rt.screen.frame
            except date.DateChanged:
                # An interruption before the AP click may re-enter opening.
                # After a charged click, follow the same save.
                if opening_submitted is not None:
                    started = started or opening_submitted()
        raise hiro_module('runtime.loop').ResumeBlocked('Hiro cultivation date recovery exhausted')
