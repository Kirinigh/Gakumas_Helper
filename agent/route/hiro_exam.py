"""Feed a pinned Hiro cultivation exam into the shared Maa policy host."""

from dataclasses import replace

from agent.live_decision.hiro_runtime import hiro_module, hiro_binding


def update_exam_entry(entry, label, session):
    produce = session.state.produce or {}
    stats = produce.get('params') or {}
    parameters = tuple(stats.get(attr) for attr in ('Vo', 'Da', 'Vi'))
    stamina = produce.get('max_stamina')
    phase, _, detail = label.partition('#')
    lesson = phase.startswith('hajime-lesson-')
    if lesson:
        _, kind = hiro_module('runtime.hajime').lesson_kind(phase)
        stage = None
    else:
        stage = phase.rsplit('-', 1)[-1]
        kind = None
    return replace(entry,
        parameters=parameters if all(type(value) is int and value > 0 for value in parameters) else entry.parameters,
        max_stamina=stamina if type(stamina) is int and stamina > 0 else entry.max_stamina,
        round_key=stage,
        audition_number=int(detail.removeprefix('audition-')) if detail.startswith('audition-') else None,
        lesson_kind=kind)


def recover_exam_interrupt(rt, frame):
    """Resume the current save after Hiro's title-returning interruptions."""
    if not frame.has_dialog:
        return False
    interrupts = hiro_module('runtime.interrupts')
    date = hiro_module('loop.date_change').DateChanged
    try:
        interrupts.date_changed(frame, rt)
        message = hiro_module('loop.comm_error').comm_error(frame.boxes)
        if message and message['kind'] != 'retry':
            interrupts.comm(frame, rt)
    except date:
        with hiro_binding(rt):
            try:
                hiro_module('runtime.play').recover_date_change(rt, rt.session)
                hiro_module('runtime.finish').resume_produce(rt)
            except hiro_module('loop.date_change').DataLinkBlocked as error:
                raise hiro_module('runtime.loop').ResumeBlocked(str(error)) from error
        return True
    return False


def _during_exam_capture(rt, capture):
    frame_type = hiro_module('runtime.frame').Frame

    def read():
        image = capture()
        if recover_exam_interrupt(rt, frame_type(image, rt.clock())):
            return capture()
        return image

    return read


def play_cultivation_exam(context, entry, algorithm, *, log, battle_type=None):
    from custom.action.produce import _open_idol_road_maa
    from custom.action.live_decision import _ocr_live_score, _locate_live_self_rank

    from agent.live_decision.exam_loop import ExamSettlement
    from agent.live_decision.hiro_maa_exam import run_maa_screen_exam
    from agent.live_decision.cultivation_maa_exam import CultivationMaaExam, read_lesson_result

    def read_score(ctx, image):
        result = _ocr_live_score(ctx, image, _locate_live_self_rank(ctx, image))
        return result.get('value') if result.get('accepted') else None

    def play(rt, label, session, ended):
        exam_entry = update_exam_entry(entry, label, session)

        def factory(ctx, verify, *, log, client_size):
            if battle_type is not None and not label.startswith('lesson-'):
                return battle_type(ctx, verify, log=log, entry=exam_entry, algorithm=algorithm,
                                   client_size=client_size, score_reader=read_score, label=label, session=session)
            return CultivationMaaExam(ctx, verify, log=log, entry=exam_entry, algorithm=algorithm,
                                      client_size=client_size, score_reader=read_score)

        battle, _ = _open_idol_road_maa(context, log, factory=factory)
        if label.startswith('hajime-lesson-'):
            battle.hiro_lesson_ended = ended
        battle.capture = _during_exam_capture(rt, battle.capture)
        resume_blocked = hiro_module('runtime.loop').ResumeBlocked
        resume_exam = getattr(rt, 'gkh_resume_exam', False)
        recommended = algorithm == 'recommended'
        try:
            if resume_exam or recommended:
                def read_result(image):
                    if label.startswith('hajime-lesson-'):
                        if not ended(image):
                            return None
                        attr, _ = hiro_module('runtime.hajime').lesson_kind(label)
                        key = {'Vo': 'vocal', 'Da': 'dance', 'Vi': 'visual'}.get(attr)
                        result = read_lesson_result(battle.ocr(image), key) if key else None
                        return ExamSettlement(None, result['score'] if result else None, None)
                    if label.startswith('lesson-'):
                        return ExamSettlement(None, None, None) if ended(image) else None
                    if not battle.result_page_ready(image):
                        return None
                    result = battle.read_result_page(image)
                    return ExamSettlement(result['rank'], result['score'], None,
                                          failed=result.get('failed', False)) if result is not None else None

                result = run_maa_screen_exam(battle, read_result, label=label, session=session,
                                             capture=battle.capture, classify=rt.classify,
                                             policy='recommended' if recommended else 'hiro-screen')
                rt.gkh_resume_exam = False
            else:
                result = battle.run_cultivation(hiro_session=session)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            # The shared executor can have submitted a click. An ordinary
            # Runtime handler exception would rerun the whole exam.
            raise resume_blocked(str(error)) from error
        if result is None:
            raise resume_blocked('Maa exam returned without a settlement')
        log(dict(event='cultivation_exam_result', produce=entry.produce, idol=entry.idol, label=label,
                 rank=result.rank, score=result.score, failed=getattr(result, 'failed', False),
                 algorithm='hiro-screen' if resume_exam and not recommended else algorithm))
        return result

    return play
