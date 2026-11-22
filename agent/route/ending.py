"""Read HIF result/reward pages after this session's accepted round receipt."""
from __future__ import annotations

import re
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path

import cv2
from utils.runtime_paths import resolve_runtime_path

from .hif import SUPPORTED_IDOL, active_hif_role
from .ownership import valid_terminal_frame_ids

SUMMARY_DESTINATIONS = (
    'ending_summary', 'ending_performance', 'ending_photo', 'ending_generation', 'ending_complete', 'ending_home')
ROUND_TWO_RESULT_DESTINATIONS = (*SUMMARY_DESTINATIONS, 'ending_exit_confirmation')
DEFEAT_EXIT_DESTINATIONS = (*SUMMARY_DESTINATIONS, 'ending_failure')
STAR_REWARD_DESTINATIONS = SUMMARY_DESTINATIONS
FINAL_EVALUATION_DESTINATIONS = SUMMARY_DESTINATIONS
MEMORY_GENERATION_DESTINATIONS = ('ending_generated_memory', 'ending_complete', 'ending_home')
MEMORY_COMPLETION_DESTINATIONS = SUMMARY_DESTINATIONS
MEMORY_DETAIL_DESTINATIONS = SUMMARY_DESTINATIONS
PRODUCE_COMPLETION_DESTINATIONS = SUMMARY_DESTINATIONS
ROUND_ONE_RESULT_DESTINATIONS = ('interval', 'ending_summary')


def same_closing_page(source, observed, current):
    """Recognize repeated award and evaluation screens before the memory is generated."""
    if source.scene == observed.scene == 'round_one_award':
        return source.page_id == observed.page_id
    return (source.scene == 'ending_summary' and current.get('summary_kind') == 'final_evaluation'
        and source.page_id == observed.page_id)


def _display_row(rows, name, pattern, roi, evidence, *, label=None, context=None, image=None, node=None):
    from .production import _row, _text
    try:
        if label is not None:
            _text(rows, *label)
    except ValueError as error:
        evidence.setdefault('display_warnings', {})[name] = str(error)
        return None
    try:
        observed = _row(rows, pattern, roi)
    except ValueError as error:
        if node is None or context is None:
            evidence.setdefault('display_warnings', {})[name] = str(error)
            return None
        if getattr(image, 'shape', ()) != (1280, 720, 3):
            raise ValueError('unsupported completion display crop resolution')
        from live_decision.hif_recovery import _field
        try:
            observed = _field(context, image, node, roi, pattern)
        except ValueError as error:
            evidence.setdefault('display_warnings', {})[name] = str(error)
            return None
    evidence[name] = observed
    return observed


def star_reward_destinations(round_number):
    if type(round_number) is not int or round_number not in (1, 2):
        raise ValueError('unsupported HIF star reward round')
    return ('interval', 'round_one_award', *STAR_REWARD_DESTINATIONS) if round_number == 1 else STAR_REWARD_DESTINATIONS


def read_round_one_award(context, image, rows):
    """Acknowledge a known obtained panel; inventory owns its business identity."""
    import csv
    from pathlib import Path

    from utils.runtime_paths import resolve_runtime_path

    from .calendar import read_route_resources
    from .production import _text
    from .activity_production import canonical, receipt_panel, receipt_title
    receipt_panel(image)
    names = set()
    for filename in ('p_items.csv', 'p_drinks.csv', 'skill_cards.csv'):
        path = resolve_runtime_path(Path('assets/data')/filename, Path('data')/filename)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            names.update(canonical(row['name']) for row in csv.DictReader(stream))
    # This acknowledges an already obtained page, not a business identity or
    # new reward choice. Use the existing free-caption tier for its complete
    # catalogued title. The later inventory observation owns the resource value.
    title = receipt_title(context, image, rows, tuple(sorted(names)), minimum=.85)
    _text(rows, r'H[.]I[.]F本戦', (20, 20, 190, 80))
    body = [row for row in rows if 75 <= row['box'][0] and 905 <= row['box'][1]
        and row['box'][0]+row['box'][2] <= 645 and row['box'][1]+row['box'][3] <= 1060]
    lines = []
    for row in sorted(body, key=lambda item: item['box'][1]+item['box'][3]/2):
        center = row['box'][1]+row['box'][3]/2
        if not lines or abs(center-lines[-1][0]) > 12:
            lines.append((center, [row]))
        else:
            lines[-1][1].append(row)
    description = None
    if body and all(row['score'] >= .95 for row in body):
        description = ''.join(canonical(row['text']) for _, group in lines
            for row in sorted(group, key=lambda item: item['box'][0]))
    # The known complete title and settled award panel permit dismissal only.
    # No effect, upgrade, P-item count or permanent card is inferred here.
    return dict(read_route_resources(rows), phase='round_one_award', award_name=canonical(title['text']),
        award_description=description, target=tuple(title['box']),
        reward_evidence=dict(title=title, description=body))


def ending_round_context(receipt, *, round_number=2):
    from live_decision.hif_session import HifRoundContext
    raw = receipt.get('round_context')
    if not isinstance(raw, dict) or set(raw) != {
            'run_id', 'revision', 'round_number', 'base_turn_count', 'frame_ids'}:
        raise ValueError('HIF ending lacks the current notebook round context')
    context = HifRoundContext(**raw)
    if ((context.run_id, context.revision, context.round_number)
            != (receipt['run_id'], receipt['revision'], round_number)
            or set(context.frame_ids) & set(receipt['frame_ids'])):
        raise ValueError('HIF ending notebook proof differs from its terminal receipt')
    return context


def verified_ending_receipt(host, *, terminal_report=None, round_number=2):
    if type(round_number) is not int or round_number not in (1, 2):
        raise ValueError('unsupported HIF terminal recovery round')
    handoff = host.handoff or {}
    receipt, live = handoff.get('terminal_receipt'), handoff.get('live_context')
    if (host.ownership is None or host.ownership['owner'] != 'task600'
            or host.ownership['phase'] != 'recovery' or handoff.get('round_number') != round_number
            or (handoff.get('idol'), handoff.get('plan')) != active_hif_role(host)
            or not isinstance(receipt, dict) or not isinstance(live, dict)
            or receipt.get('accepted') is not True or receipt.get('terminal') is not True
            or receipt.get('round_number') != round_number or live.get('round_number') != round_number
            or (receipt.get('run_id'), receipt.get('revision')) != (live.get('run_id'), live.get('revision'))
            or type(receipt.get('score')) is not int or receipt['score'] < 0
            or not valid_terminal_frame_ids(receipt.get('frame_ids'))):
        raise ValueError('HIF ending requires this session accepted terminal receipt for the current round')
    if terminal_report is not None:
        if not isinstance(terminal_report, dict):
            raise ValueError('HIF closing source is not a native result record')
        opening = terminal_report.get('handoff_opening', {})
        returned = terminal_report.get('returned_to_nonbattle') is True
        if not returned and terminal_report.get('handoff_return_prepared') is True:
            from .terminal_source import verify_bound_terminal_return
            verify_bound_terminal_return(run_id=host.run_id, ownership=host.ownership,
                handoff=handoff, report=terminal_report)
            returned = True
        if (not isinstance(opening, dict) or not isinstance(handoff.get('handoff_id'), str)
                or not handoff['handoff_id'].strip()
                or terminal_report.get('status') != 'TERMINAL_VERIFIED'
                or not returned
                or terminal_report.get('terminal_settlement') != receipt
                or terminal_report.get('endpoints', {}).get('game', {}).get('pid') != host.game_pid
                or opening.get('route_run_id') != host.run_id
                or opening.get('handoff_id') != handoff.get('handoff_id')
                or opening.get('live_run_id') != receipt['run_id']
                or opening.get('live_revision') != receipt['revision']
                or opening.get('round_number') != round_number):
            raise ValueError('HIF closing source differs from the current returned terminal receipt')
        receipt = deepcopy(receipt)
        receipt['round_context'] = deepcopy(terminal_report.get('round_context'))
    ending_round_context(receipt, round_number=round_number)
    return receipt


def read_verified_round_one_result(rows, receipt, *, role=(SUPPORTED_IDOL, 'sense')):
    from live_decision.hif_result import _read_hif_result
    from live_decision.hif_profiles import supported_hif_identity

    from .production import UnknownRecoveryPage, _row
    profile = supported_hif_identity(*role)
    round_context = ending_round_context(receipt, round_number=1)
    try:
        result = _read_hif_result([SimpleNamespace(**row) for row in rows], idol=profile.idol,
            round_context=round_context)
    except ValueError as error:
        raise UnknownRecoveryPage(str(error)) from error
    if result['score'] != receipt['score']:
        raise ValueError('current round-one result differs from the accepted terminal score')
    try:
        next_page = _row(rows, r'次へ', (220, 1080, 310, 130))
    except ValueError as error:
        raise UnknownRecoveryPage(str(error)) from error
    return dict(phase='round_one_result', target=tuple(next_page['box']), score=result['score'])


def read_verified_round_two_result(rows, receipt, *, role=(SUPPORTED_IDOL, 'sense')):
    from live_decision.hif_result import _read_hif_result
    from live_decision.hif_profiles import supported_hif_identity

    from .production import _row
    profile = supported_hif_identity(*role)
    result = _read_hif_result([SimpleNamespace(**row) for row in rows],
        idol=profile.idol, round_context=ending_round_context(receipt))
    if result['score'] != receipt['score']:
        raise ValueError('current result contradicts this session accepted terminal score')
    if receipt.get('cumulative_score') is not None and result.get('cumulative_score') != receipt['cumulative_score']:
        raise ValueError('current cumulative result contradicts its accepted terminal receipt')
    if receipt.get('outcome') is not None and result.get('outcome') != receipt['outcome']:
        raise ValueError('current result outcome contradicts its accepted terminal receipt')
    next_page = _row(rows, r'次へ', (220, 1080, 310, 130))
    return dict(phase='end_result', target=tuple(next_page['box']), score=result['score'],
        cumulative_score=result.get('cumulative_score'), outcome=result.get('outcome'))


def read_defeat_exit_confirmation(rows, receipt):
    from .production import _row, _text
    if receipt.get('outcome') != 'defeat':
        raise ValueError('ending exit confirmation requires the accepted defeat result')
    _text(rows, r'再挑戦確認', (25, 810, 650, 95))
    _text(rows, r'再挑戦が可能ですが本当に終了しますか[?？]', (25, 910, 670, 80))
    _text(rows, r'プロデュース終了を選択した場合は失敗となります', (25, 970, 670, 65))
    _text(rows, r'再挑戦', (420, 1115, 200, 85))
    # The actual left label wraps into two lines. Its complete unique final
    # word plus the explicit full dialog/warning identify the normal end action;
    # a weak first-line OCR result is not promoted or used as a target.
    end = _row(rows, r'終了', (160, 1130, 200, 85))
    return dict(phase='ending_exit_confirmation', target=tuple(end['box']), outcome='defeat')


def read_star_reward(rows, *, context=None, image=None):
    from .production import _row, _text, read_continue_caption
    _text(rows, r'スター性獲得', (120, 330, 500, 150))
    increment = int(_text(rows, r'\+\d{1,4}', (220, 440, 280, 110))[1:])
    try:
        full_total = _row(rows, r'\d{1,5}', (180, 700, 380, 200))
    except ValueError:
        full_total = None
    if context is not None:
        from live_decision.hif_recovery import _field
        if getattr(image, 'shape', ()) != (1280, 720, 3):
            raise ValueError('unsupported star reward crop resolution')
        # Actual 123 merged the orange star icon into the full-frame number.
        # Independently recognise only the complete digits, at the same .95
        # gate; never strip an OCR star or reuse the earlier observed 1103.
        current_total = _field(context, image, 'EndingStarTotal', (320, 760, 195, 100), r'\d{1,5}')
        if full_total is not None and current_total['text'] != full_total['text']:
            raise ValueError('star reward numeral crop contradicts the strong full-frame value')
    elif full_total is not None:
        current_total = full_total
    else:
        raise ValueError('HIF production field is not uniquely observed: star total')
    total = int(current_total['text'])
    if total < increment:
        raise ValueError('current star reward total is smaller than its increment')
    tap = read_continue_caption(rows, (280, 1170, 180, 110))
    return dict(phase='ending_summary', summary_kind='star_reward', target=tuple(tap['box']),
        star_total=total, star_increment=increment, reward_evidence=dict(total=current_total,
            full_frame_total=full_total, full_frame_candidates=[row for row in rows
                if 180 <= row['box'][0] and 700 <= row['box'][1]
                and row['box'][0]+row['box'][2] <= 560 and row['box'][1]+row['box'][3] <= 900]))


def read_final_evaluation(rows):
    from .production import _text, read_continue_caption
    _text(rows, r'最終プロデュース評価', (50, 130, 620, 150))
    # The accepted terminal and full evaluation heading identify this page.
    # Decorative logo text does not control the original free continuation.
    evidence = {}
    total = _display_row(rows, 'evaluation_total', r'\d{1,6}', (190, 780, 360, 160), evidence)
    tap = read_continue_caption(rows, (240, 1080, 250, 130))
    return dict(phase='ending_summary', summary_kind='final_evaluation', target=tuple(tap['box']),
        evaluation_total=int(total['text']) if total is not None else None, reward_evidence=evidence)


def read_memory_generation(rows):
    from .production import _row, _text
    _text(rows, r'MEMORY', (280, 150, 170, 80))
    generate = _row(rows, r'生成', (280, 910, 170, 120))
    return dict(phase='ending_generation', target=tuple(generate['box']))


def read_generated_memory(rows):
    from .production import _row, _text
    complete = _row(rows, r'メモリー生成完了', (50, 180, 620, 160))
    _text(rows, r'HIF', (350, 520, 280, 220))
    # The original completed-page template is itself clicked. Its full-page
    # continue prompt blinks; use the current complete heading, not old pixels.
    return dict(phase='ending_generated_memory', memory_brand='HIF', target=tuple(complete['box']))


def read_generated_memory_detail(rows, *, role=(SUPPORTED_IDOL, 'sense')):
    from live_decision.hif_profiles import supported_hif_identity

    from .production import _row, _text
    profile = supported_hif_identity(*role)
    _text(rows, r'HIF', (470, 110, 200, 110))
    _text(rows, re.escape(profile.song), (90, 460, 250, 70))
    _text(rows, re.escape(profile.idol), (90, 500, 230, 85))
    next_page = _row(rows, r'次へ', (270, 1100, 190, 110))
    return dict(phase='ending_complete', memory_kind='generated_detail',
        memory_brand='HIF', memory_idol=profile.identity, target=tuple(next_page['box']))


def _completion_next(rows, image):
    if getattr(image, 'shape', ()) == (1280, 720, 3):
        path = Path('resource/base/image/next.png')
        reference = cv2.imread(str(resolve_runtime_path(Path('assets') / path, path)))
        if reference is not None:
            # hiro searches the whole normalized frame and uses its strongest match.
            scores = cv2.matchTemplate(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            if score >= .8:
                return (x, y, reference.shape[1], reference.shape[0])
    for row in rows:
        if row['text'] == '次へ' and row['box'][1] + row['box'][3] / 2 > 700:
            return tuple(row['box'])
    raise ValueError('current produce completion has no next control')


def read_produce_completion(rows, *, context=None, image=None):
    from .production import _text
    _text(rows, r'プロデュース', (380, 0, 320, 85))
    _text(rows, r'完了', (460, 45, 160, 90))
    display = {}
    level = _display_row(rows, 'completion_level', r'\d{1,3}', (380, 130, 150, 60), display,
        label=(r'PLV', (360, 90, 110, 75)))
    fan_count = _display_row(rows, 'completion_fans', r'\d{1,3}(?:,\d{3})*人', (380, 260, 330, 65), display,
        label=(r'ファン数', (360, 220, 200, 65)), context=context, image=image, node='EndingFanCount')
    stage = None
    evidence = None
    if any(row['score'] >= .85 and row['text'].startswith('アチーブメント進') for row in rows):
        # Keep the optional label when the distinctive word was actually read.
        if context is not None and getattr(image, 'shape', ()) == (1280, 720, 3):
            evidence = _display_row(rows, 'achievement_heading', r'アチーブメント',
                (385, 520, 195, 45), display, context=context, image=image, node='EndingAchievementHeading')
        else:
            evidence = _display_row(rows, 'achievement_heading', r'アチーブメント進捗', (385, 520, 270, 45), display)
        if evidence is not None:
            stage = 'achievements'
    elif _display_row(rows, 'item_heading', r'獲得アイテム', (360, 500, 250, 90), display) is not None:
        stage = 'items'
    return dict(phase='ending_summary', summary_kind='produce_complete',
        completion_stage=stage, completion_level=int(level['text']) if level is not None else None,
        completion_fans=int(fan_count['text'][:-1].replace(',', '')) if fan_count is not None else None,
        reward_evidence=dict(display, achievement_heading=evidence, fan_count=fan_count), target=_completion_next(rows, image))


def read_received_event_rewards(rows):
    """Dismiss rewards already received during the authorised closing flow."""
    from .production import _row, _text
    _text(rows, r'受取完了', (25, 300, 250, 100))
    close = _row(rows, r'閉じる', (290, 1100, 180, 100))
    return dict(phase='ending_summary', summary_kind='received_event_rewards', target=tuple(close['box']))


def read_event_points_result(rows):
    """Continue the actual event-point result after the received popup."""
    from .production import _row, _text
    _text(rows, r'イベントPt', (240, 160, 260, 95))
    _text(rows, r'獲得イベントPt', (20, 230, 230, 80))
    _text(rows, r'累計イベントPt', (60, 330, 260, 85))
    next_page = _row(rows, r'次へ', (300, 1080, 170, 130))
    return dict(phase='ending_summary', summary_kind='event_points_result', target=tuple(next_page['box']))


def read_ending_page(context, image, rows, host, *, terminal_report=None, round_number=2):
    from live_decision.hif_profiles import supported_hif_identity

    receipt = verified_ending_receipt(host, terminal_report=terminal_report, round_number=round_number)
    role = active_hif_role(host)
    if any(row['score'] >= .95 and row['text'] == '受取完了' for row in rows):
        return read_received_event_rewards(rows)
    if any(row['score'] >= .95 and row['text'] == '獲得イベントPt' and 230 <= row['box'][1] <= 300 for row in rows):
        return read_event_points_result(rows)
    if round_number == 1:
        from .activity_production import receipt_panel
        try:
            receipt_panel(image)
        except ValueError:
            pass
        else:
            return read_round_one_award(context, image, rows)
    if any(row['score'] >= .85 and row['text'] == '再挑戦確認' for row in rows):
        return read_defeat_exit_confirmation(rows, receipt)
    if any(row['score'] >= .85 and row['text'] == 'スター性獲得' for row in rows):
        return read_star_reward(rows, context=context, image=image)
    if any(row['score'] >= .85 and row['text'] == '最終プロデュース評価' for row in rows):
        return read_final_evaluation(rows)
    if any(row['score'] >= .85 and row['text'] == 'メモリー生成完了' for row in rows):
        return read_generated_memory(rows)
    if any(row['score'] >= .85 and row['text'] == 'MEMORY' for row in rows):
        # verified_ending_receipt bound this terminal to the same run and idol.
        return dict(read_memory_generation(rows), generation_idol=host.handoff['idol'],
            generation_idol_source='terminal_handoff')
    if any(row['score'] >= .85 and row['text'] == '完了'
            and 460 <= row['box'][0] and 45 <= row['box'][1]
            and row['box'][0] + row['box'][2] <= 620 and row['box'][1] + row['box'][3] <= 135 for row in rows):
        return read_produce_completion(rows, context=context, image=image)
    if any(row['score'] >= .85 and row['text'] == supported_hif_identity(*role).song
            and 460 <= row['box'][1] < 530 for row in rows):
        return read_generated_memory_detail(rows, role=role)
    # Other closing pages require their own current page identity. A generic
    # Next, portrait orientation or an upstream Click_1 is insufficient.
    try:
        return (read_verified_round_one_result(rows, receipt, role=role) if round_number == 1
            else read_verified_round_two_result(rows, receipt, role=role))
    except ValueError as result_error:
        from .home import read_active_home
        try:
            home = read_active_home(image, rows, context=context)
        except ValueError as home_error:
            raise ValueError(f'HIF closing page is not identified: {result_error}; {home_error}') from home_error
        return dict(phase='ending_home', target=None, ap=home['ap'], max_ap=home['max_ap'], home_evidence=home)
