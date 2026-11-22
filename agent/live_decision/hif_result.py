"""Read-only HIF per-round result verification; never advances or retries Live."""

import re
import math
import time
import unicodedata
from itertools import count

from .inventory import _inventory_ocr
from .hif_session import resolve_hif_round_context
from .hif_profiles import supported_hif_identity


class _ResultNotReady(ValueError):
    pass


def _text(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', value))


def read_hif_result_score(rows, *, idol, round_context=None):
    return _read_hif_result(rows, idol=idol, round_context=round_context)['score']


def _read_hif_result(rows, *, idol, round_context=None):
    """Bind the result score to the named player's row, never to NPC rank.

    Layout reference: pinned Hiro v2.3.3 ProduceHIFReChallengeAuto, title
    ROI 0/70/420/80 and score list 0/250/720/510. No retry behavior is reused.
    """
    round_number, _ = resolve_hif_round_context(round_context)
    rows = tuple(rows)
    valid = [row for row in rows if math.isfinite(row.score) and .95 <= row.score <= 1]
    headings = [_text(row.text) for row in valid
                if 0 <= row.box[0] < 420 and 60 <= row.box[1] < 150]
    heading = ''.join(headings)
    # Native R2 victory/defeat replaces the round heading with total + increment.
    # Its round must come from the bound notebook evidence, never a default.
    outcomes = [value for value in headings if value in ('敗退', '優勝')]
    ranked_result = bool(outcomes)
    heading_error = 'HIF result heading does not confirm its explicit round'
    if ranked_result:
        ranks = [row for row in valid if _text(row.text) == '現在順位'
                 and 0 <= row.box[0] < 200 and 150 <= row.box[1] < 220]
        if (round_context is None or round_number != 2 or len(outcomes) != 1
                or '本戦' in heading or 'ラウンド' in heading or len(ranks) > 1):
            raise ValueError(heading_error)
        if not ranks:
            raise _ResultNotReady(heading_error)
    else:
        if any(f'ラウンド{other}結果' in heading for other in (1, 2) if other != round_number):
            raise ValueError(heading_error)
        if '本戦' not in heading or f'ラウンド{round_number}結果' not in heading:
            raise _ResultNotReady(heading_error)
    # 1576's first ranked panel starts at y=225; its player name starts at 245.
    # Keep the legacy round-result list unchanged and bind by name at every rank.
    name_top = 225 if ranked_result else 250
    names = [row for row in valid if _text(row.text) == idol
             and name_top <= row.box[1] < 760
             and (not ranked_result or 200 <= row.box[0] < 600)]
    if not names:
        raise _ResultNotReady('HIF result player row is not unique')
    if len(names) != 1:
        raise ValueError('HIF result player row is not unique')
    name = names[0]
    x,y,w,h = name.box
    scores = []
    number = r'(?:\d+|\d{1,3}(?:[,\.]\d{3})+)'
    for row in rows:
        rx,ry,_,rh = row.box
        inline = rx >= x+w and abs(ry+rh/2-y-h/2) <= 70
        below = abs(rx-x) <= 10 and 0 <= ry-(y+h) <= 70
        if not (inline or below) or not 250 <= ry < 760:
            continue
        value = _text(row.text)
        pattern = f'({number})\\+({number})' if ranked_result else number
        match = re.fullmatch(pattern, value)
        if not match:
            continue
        if not math.isfinite(row.score) or not .95 <= row.score <= 1:
            raise _ResultNotReady('HIF player result score is low confidence')
        if ranked_result:
            total, score = (int(part.replace(',', '').replace('.', '')) for part in match.groups())
            if total < score:
                raise ValueError('HIF cumulative result is below its round increment')
            scores.append(dict(score=score, cumulative_score=total,
                               outcome='victory' if outcomes[0] == '優勝' else 'defeat'))
        else:
            scores.append(dict(score=int(value.replace(',','').replace('.',''))))
    if not scores:
        raise _ResultNotReady('HIF player result score is not unique')
    if len(scores) != 1:
        raise ValueError('HIF player result score is not unique')
    return scores[0]


def hif_round_one_registered_score(raw_score):
    """Round-one placement uses floor(raw score * 1.2), irrespective of rank."""
    return hif_registered_score(raw_score, round_number=1)


def hif_registered_score(raw_score, *, round_number):
    """Per-round registration, not final combined score or star-quality rewards.

    Fixed rule reference: wikiwiki.jp/gakumas/HIF/基本情報, 本戦＆インターバル.
    Round one is also verified by native result 712 (36138 -> 43365).
    """
    if type(raw_score) is not int or raw_score < 0:
        raise ValueError('HIF raw terminal score is invalid')
    if type(round_number) is not int or round_number not in (1, 2):
        raise ValueError('HIF result round is unsupported')
    return raw_score * 6 // 5 if round_number == 1 else raw_score


def read_displayed_terminal_frame(context, frame, expected, *, frame_identifier, round_context):
    """Read the result itself; a predicted ongoing branch supplies no final score."""
    run = context.gkh_hif_live_context
    round_number, _ = resolve_hif_round_context(round_context, run_id=run.run_id, revision=run.revision)
    profile = supported_hif_identity(run.idol, run.plan)
    if getattr(frame, 'shape', None) != (1280, 720, 3):
        raise ValueError('HIF result requires a calibrated frame')
    rows = _inventory_ocr(context, frame, 'HifRoundResult', roi=[0, 60, 720, 700])
    reading = _read_hif_result(rows, idol=profile.idol, round_context=round_context)
    raw_scores = sorted({state.score for state in expected if state.terminal
                         and not state.pending_choice and not state.resolution_stack})
    expected_scores = sorted({hif_registered_score(score, round_number=round_number) for score in raw_scores})
    score = reading['score']
    event = dict(terminal=True, accepted=True, input_click_count=0, **reading,
                 expected_scores=expected_scores, raw_expected_scores=raw_scores,
                 score_rule='hif-round1-floor-six-fifths' if round_number == 1 else 'hif-round2-raw',
                 frame_ids=[frame_identifier(frame)],
                 run_id=run.run_id, revision=run.revision)
    if not expected_scores:
        event['model_predicted_terminal'] = False
    elif score not in expected_scores:
        event['observed_minus_expected_scores'] = [score - predicted for predicted in expected_scores]
    if round_context is not None:
        event['round_number'] = round_number
    return event


def observe_hif_terminal_result(context, expected, *, capture_frame, frame_identifier,
                                round_context=None):
    """Wait for a readable result at an expected end of battle."""
    event = dict(terminal=True, accepted=False, input_click_count=0)
    started = time.perf_counter()
    poll_progress = getattr(context, 'gkh_hif_poll_battle_progress', None)
    run = context.gkh_hif_live_context
    resolve_hif_round_context(round_context, run_id=run.run_id, revision=run.revision)
    for attempt in count(1):
        frame = capture_frame()
        try:
            return read_displayed_terminal_frame(context, frame, expected, frame_identifier=frame_identifier,
                                                  round_context=round_context)
        except ValueError as error:
            event['reason'] = str(error)
            event['attempts'] = attempt
            if isinstance(error, _ResultNotReady) and callable(poll_progress):
                poll_progress(frame)
            elif attempt >= 20 or time.perf_counter()-started >= 20:
                break
            time.sleep(.25)
    return event


def try_displayed_terminal_frame(context, frame, expected, *, frame_identifier, observed_battle_exit=False):
    """A non-playable screen can be an animation, another panel or the actual result."""
    round_context = getattr(context, 'gkh_hif_round_context', None)
    if round_context is None:
        return None
    if not observed_battle_exit:
        # A still-playable hand does not need result OCR.
        from .hif_upstream_ui import _NODES

        recognize = getattr(context, 'run_recognition', None)
        if not callable(recognize):
            return None
        for node in ('ProduceRecognitionSkipRound', 'ProduceRecognitionHealthFlag'):
            match = recognize(node, frame, pipeline_override={node: _NODES[node]})
            if match and match.hit:
                return None
    try:
        return read_displayed_terminal_frame(context, frame, expected, frame_identifier=frame_identifier,
                                              round_context=round_context)
    except ValueError:
        return None
