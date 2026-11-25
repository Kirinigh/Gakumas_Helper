"""Hiro's HIF exam labels and result pages over the existing Maa exam host."""

import time
from copy import deepcopy
from dataclasses import replace

from agent.live_decision.hiro_runtime import hiro_module
from agent.live_decision.cultivation_maa_exam import CultivationEntry, CultivationMaaExam

from .hiro_exam import play_cultivation_exam, recover_exam_interrupt
from .qualifier import Screen
from .qualifier_result import read_qualifier_result

_HIF_STAGES = {
    'qualify-1': ('produce-007', 'mid1'),
    'qualify-2': ('produce-007', 'mid2'),
    'qualify-3': ('produce-007', 'final'),
    'r1': ('produce-008', 'mid1'),
    'r2': ('produce-008', 'final'),
}


class HifMaaExam(CultivationMaaExam):
    def __init__(self, context, verify_input, *, entry, label, session, **kwargs):
        produce, phase = _HIF_STAGES[label]
        super().__init__(context, verify_input, entry=replace(entry, produce=produce, round_key=phase), **kwargs)
        self.label, self.session = label, session

    def bind_opening(self, image):
        exam = super().bind_opening(image)
        if exam is None:
            return None
        deck = (self.session.state.produce or {}).get('deck') or []
        local = []
        for card in deck:
            candidates = self.native_rules.cards.get((card.get('card_id'), card.get('upgrade') or 0), ())
            if len(candidates) == 1:
                local.append(candidates[0])
        self.deck_ids = tuple(local)
        return exam

    def read_lesson_goal(self, image):
        return None

    def bind_audition_opening(self, image, turns, tables, upstream, audition_row):
        exam, bonus, row = super().bind_audition_opening(image, turns, tables, upstream, audition_row)
        # R2 uses the final master row, while both UINet encoders use Hiro's
        # AUDITION_MID2 label for this mode's second round.
        fields = hiro_module('bench_bridge.screen').exam_fields(
            self.label, idol=self.entry.idol, plan=None, crit=exam['crit'], produce=self.entry.produce)
        return fields, bonus, row

    def _audition_attribute(self, image):
        return hiro_module('vision.hif')._round_attr(image).get('attr')

    def read_exam_position(self, image):
        bars = hiro_module('vision.hif').read_ranking(image)
        score = bars.get('my_score')
        if score is None:
            score = self.score_reader(self.context, image)
        return self._audition_attribute(image), score

    def read_result_page(self, image):
        if self.label.startswith('qualify-'):
            rows = [dict(text=row.text, box=row.box, score=row.score) for row in self.ocr(image)]
            result = read_qualifier_result(Screen(image, rows, time.monotonic()))
            if result is not None and result['exam'] == int(self.label[-1]):
                return dict(result, rank=None)
        else:
            from .hiro_hif import _read_ocr

            rows = _read_ocr(self.context, image)
            boxes = [(row.text, (x, y, x + w, y + h))
                     for row in rows for x, y, w, h in (row.box,)]
            result = hiro_module('vision.pages').read_finals_ranking_page(image, boxes)
            wanted = 'round_result' if self.label == 'r1' else 'final'
            if result.get('kind') == wanted:
                score = result.get('round_score')
                if score is not None:
                    return dict(score=score, rank=result.get('rank'))
        return None

    def exam_label(self, exam):
        return self.label


def play_hif_exam(context, device, idol, *, log, play_finals=None):
    """Dispatch selection exams and the chosen finals policy."""
    from custom.action.produce import ProduceCardsAuto

    battle = play_cultivation_exam(context, CultivationEntry('produce-007', idol, None),
                                  'uinet-2.1', log=log, battle_type=HifMaaExam)

    def play(rt, label, session, ended):
        produce = session.state.produce or {}
        log(dict(event='hiro_exam_deck', label=label, segment=session.trace.mode,
                 through_step=session.trace.seq, cards=deepcopy(produce.get('deck'))))
        if label.startswith('lesson-'):
            # HIF public courses retain the existing in-game recommended
            # policy; qualifier and finals grades use the UINet factory.
            if not ProduceCardsAuto()._run_recommended_exam(context, authorize=device.authorise):
                raise hiro_module('runtime.loop').ResumeBlocked('HIF public course result was unreadable')
            return
        if play_finals is not None and label in ('r1', 'r2'):
            play_finals(rt, int(label[-1]))
            return
        battle(rt, label, session, ended)

    return play


def play_formal_hif_finals(context, *, path, runtime, run_id, game, profile, authorize, log):
    """Reuse the HIF factory, observations, execution and durable action receipts."""
    from live_decision.hif_battle_entry import run_current_page_battle

    completed = {1: 0, 2: 0}

    def current_task():
        authorize()
        return True

    def play(rt, number):
        # Hiro owns the in-game retry decision and reaches another opening
        # before calling back. Only that new exam gets a new receipt identity.
        attempt = completed[number] + 1
        battle_run = f'{run_id}-r{number}-exam{attempt}'
        recoveries = hiro_module('runtime.play').MAX_DATE_CHANGES
        blocked = hiro_module('runtime.loop').ResumeBlocked
        for recovery in range(recoveries + 1):
            try:
                result = run_current_page_battle(context, path=path, runtime=runtime,
                    run_id=battle_run, game_pid=game['pid'], client_size=tuple(game['observed_client_size']),
                    policy_profile=profile, authorise_task=current_task, expected_round=number)
            except (OSError, RuntimeError, ValueError) as error:
                result = dict(status='STOP', stop_reason=str(error))
            terminal = result.get('terminal_settlement') or {}
            log(dict(event='hiro_formal_hif_result', round_number=number, attempt=attempt, policy_profile=profile,
                status=result['status'], record=result.get('last_record'), input_count=result.get('input_count'),
                score=terminal.get('score'), stop_reason=result.get('stop_reason')))
            if (result['status'] == 'TERMINAL_VERIFIED' and result.get('current_battle_finished') is True
                    and terminal.get('round_number') == number):
                completed[number] = attempt
                return
            if recovery == recoveries or not recover_exam_interrupt(rt, rt.screen.shot()):
                raise blocked(result.get('stop_reason') or 'HIF finals not settled')

    return play
