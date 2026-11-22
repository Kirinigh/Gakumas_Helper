"""Bounded normal-UI recovery of the original HIF save; no battle reconciliation."""
from __future__ import annotations

import re
import json
import time
from copy import deepcopy
from pathlib import Path

from utils.runtime_paths import resolve_runtime_path

from .hif import active_hif_role, HifConfig
from .host import SAVED_CULTIVATION_NAVIGATION, HifRouteHost, HifRoutePage
from .engine import Target, Outcome, Decision, Observation, TransitionGate, propose_unique, source_frames_match, validate_observation
from .checkpoint import transition_checkpoint


def _business_scope(host):
    return json.loads(json.dumps(dict(run_id=host.run_id, ownership=host.ownership,
        ledger=host.ledger.record() if host.ledger is not None else None), allow_nan=False))


def _row(rows, pattern, roi, *, minimum=.95):
    x, y, width, height = roi
    found = [row for row in rows if minimum <= row['score'] <= 1 and re.fullmatch(pattern, row['text'])
        and x <= row['box'][0] and y <= row['box'][1]
        and row['box'][0] + row['box'][2] <= x + width and row['box'][1] + row['box'][3] <= y + height]
    if len(found) != 1:
        raise ValueError('saved-cultivation field is not unique: ' + pattern)
    return found[0]


def _inside_target(row, box):
    x, y, width, height = row['box']
    left, top, wide, high = box
    if not (x <= left and y <= top and left + wide <= x + width and top + high <= y + height):
        raise ValueError('reviewed navigation interior is outside its current complete label')
    return box


def read_saved_resume(image, rows, *, saved_day=7, nonbattle=False):
    from live_decision.hif_profiles import notebook_hif_identity

    from .home import _unique_home_template

    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('saved-cultivation page resolution is uncalibrated')
    if type(saved_day) is not int or not 1 <= saved_day <= 7:
        raise ValueError('saved cultivation day is not explicitly bound')
    # Both preview layouts show the same save's idol and day. The stored-data
    # notice shifts these two rows vertically.
    day = _row(rows, r'[1-7]/7日目', (500, 360, 160, 250))
    if day['text'] != f'{saved_day}/7日目':
        raise ValueError('preview day differs from the current saved cultivation')
    identity = _row(rows, r'\[.+\].+', (200, 595, 400, 260))['text']
    title, actor = identity[1:].rsplit(']', 1)
    notebook_hif_identity(title, actor)
    # fb8fd00c HIF.json ProduceHIF__ProduceChooseContinue clicks this exact
    # template (hif/produce/continue.png upstream) with Maa's full-frame,
    # 0.7 threshold and TM_CCOEFF_NORMED defaults.
    screen = (0, 0, 720, 1280)
    try:
        button, _ = _unique_home_template(image, 'produce/continue.png', screen, .7)
    except ValueError:
        # Hiro's resume_rules also reads the current dialog and button when
        # the resume template does not match after a date change.
        if not nonbattle:
            raise
        texts = [row['text'] for row in rows]
        if (not any('プロデュース再開' in text for text in texts)
                or any('破棄' in text or '開始する' in text for text in texts)):
            raise
        button = _row(rows, r'再開する', screen, minimum=0)
    return dict(scene='resume_saved', target=tuple(button['box']))


def _title_identity(rows):
    # Version is a navigation landmark, not a resource value. The independent
    # complete copyright and its click interior retain their strong gate.
    _row(rows, r'Ver[.]\d+[.]\d+[.]\d+', (15, 15, 190, 55), minimum=.85)
    return _row(rows, r'THEIDOLM@STER.*BandaiNamcoEntertainmentInc[.]?', (15, 1190, 610, 50))


def read_date_page(image, rows, *, saved_day=7, context=None, nonbattle=False, recovering_login=False):
    """Read only the known UI surfaces from this run's 21--32 recovery."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('saved-cultivation page resolution is uncalibrated')
    candidates = []
    def collect(reader):
        try:
            candidates.append(reader())
        except ValueError:
            pass

    def date_changed():
        if nonbattle:
            from .nonbattle_observation import _within

            # hiro date_change_button recognizes the forced data update in
            # the dialog strip and clicks its title button, like a date change.
            strip = (0, 880, 720, 360)
            if any(_within(row['box'], strip) and 'データ更新' in row['text'] for row in rows):
                title = _row(rows, r'タイトルへ', strip, minimum=0)
                return dict(scene='data_update', target=tuple(title['box']))
            if not any(_within(row['box'], (40, 940, 620, 165))
                    and re.search(r'日付変更|日付が変わりました|日期变更|日期變更', row['text']) for row in rows):
                raise ValueError('date change notice is absent')
            # These rows already passed the OCR recognizer's own threshold.
            title = _row(rows, r'タイトルへ|返回标题|返回標題', (225, 1115, 275, 95), minimum=0)
            # Keep the established tap stable across OCR jitter when it fits the label.
            interior = (340, 1150, 70, 10)
            target = interior if _within(interior, title['box']) else tuple(title['box'])
            return dict(scene='date_changed', target=target)
        _row(rows, r'日付変更', (35, 930, 250, 80))
        _row(rows, r'日付が変わりました[。]?', (40, 1020, 500, 80))
        # The existing bounded click_ocr navigation gate is .85. The actual
        # 21 title button is .913; both independent dialog fields remain .95.
        title = _row(rows, r'タイトルへ', (250, 1120, 250, 80), minimum=.85)
        return dict(scene='date_changed', target=_inside_target(title, (340, 1150, 70, 10)))
    collect(date_changed)

    def title():
        if nonbattle:
            # fb8 DateTitle requires both title-screen controls. Its login
            # loop taps the blank corner after checking the foreground dialogs.
            _row(rows, r'.*(?:データ連携|数据关联|資料連動).*', (20, 95, 160, 55), minimum=0)
            _row(rows, r'.*(?:プロデューサー|制作人|製作人).*', (20, 55, 410, 45), minimum=0)
            return dict(scene='game_title', target=(0, 0, 50, 50))
        copyright = _title_identity(rows)
        return dict(scene='game_title', target=_inside_target(copyright, (120, 1215, 60, 6)))
    if not nonbattle:
        collect(title)

    def login_bonus():
        from .login_display import hiro_login_notice, claimed_login_display, reject_login_controls

        if (notice := hiro_login_notice(image, rows)) is not None:
            return notice
        if nonbattle:
            reject_login_controls(rows)
            for row in rows:
                if (re.fullmatch(r'LOGIN(?:BONUS)?|BONUS', row['text'])
                        and (recovering_login or 50 <= row['box'][1] <= 130)):
                    return dict(scene='login_bonus', target=tuple(row['box']))
            if recovering_login and any(row['text'].startswith('開催期間:') for row in rows):
                page = claimed_login_display(image, rows)
                return dict(page, target=(360, 1180, 1, 1))
            raise ValueError('login bonus heading is absent')
        english = _row(rows, r'LOGINBONUS', (190, 50, 330, 80))
        reject_login_controls(rows)
        _row(rows, r'\d+日ログインまで', (350, 690, 260, 65))
        return dict(scene='login_bonus', target=_inside_target(english, (300, 82, 100, 12)))
    collect(login_bonus)

    def friend_notice():
        if nonbattle:
            from .home import _unique_home_template
            # Within a login episode the fixed upstream uses these two images.
            # A visible notice can also identify the interrupt on a normal frame.
            if recovering_login:
                try:
                    screen = (0, 0, 720, 1280)
                    _unique_home_template(image, 'friend_points.png', screen, .7)
                    close, _ = _unique_home_template(image, 'close.png', screen, .7)
                    return dict(scene='friend_notice', target=close['box'])
                except ValueError:
                    pass
            _row(rows, r'フレンドコイン獲得', (35, 390, 400, 80), minimum=0)
            close = _row(rows, r'閉じる', (300, 1120, 160, 80), minimum=0)
            return dict(scene='friend_notice', target=tuple(close['box']))
        _row(rows, r'フレンドコイン獲得', (35, 390, 400, 80))
        _row(rows, r'サポートカードとメモリーのレンタル報酬として', (70, 480, 580, 65))
        _row(rows, r'フレンドコインを\d+獲得しました[。]?', (70, 520, 500, 65))
        close = _row(rows, r'閉じる', (300, 1120, 160, 80))
        return dict(scene='friend_notice', target=_inside_target(close, (355, 1150, 25, 10)))
    collect(friend_notice)

    def saved_home():
        from .home import read_active_home
        # Opening the existing read-only save preview does not resume it.
        # The preview identifies the role and saved day before resuming.
        home = read_active_home(image, rows, preview_saved=True, context=context, optional_preview_ap=nonbattle)
        return dict(scene='saved_home', target=home['target'], evidence=home)
    collect(saved_home)

    def resume():
        return read_saved_resume(image, rows, saved_day=saved_day, nonbattle=nonbattle)
    collect(resume)

    def network():
        from .home import _unique_home_template
        from .nonbattle_observation import _within

        try:
            _row(rows, r'通信エラー', (30, 930, 300, 85))
            _row(rows, r'通信中にエラーが発生しました', (40, 1020, 500, 80))
            retry = _row(rows, r'リトライ', (440, 1120, 180, 85))
            target = _inside_target(retry, (500, 1150, 40, 10))
        except ValueError:
            if not nonbattle:
                raise
            # E209 raises the same heading above the upstream short-dialog ROI.
            _row(rows, r'通信エラー', (0, 0, 720, 1280), minimum=.85)
            # fb8 ProduceHIF__ProduceButton's retry.png search and click offset;
            # ProduceHIF__ProduceNetworkRetry identifies the right-hand control.
            button, _ = _unique_home_template(image, 'produce/retry.png', (0, 780, 720, 500), .7)
            if not _within(button['box'], (370, 1115, 265, 90)):
                raise ValueError('current network retry is outside the right-hand control')
            x, y, width, height = button['box']
            target = (x, y, width, height - 10)
        return dict(scene='network_error', target=target)
    collect(network)
    def expired():
        if nonbattle:
            if any(page['scene'] == 'network_error' for page in candidates):
                raise ValueError('current communication dialog offers retry')
            title = _row(rows, r'通信[エ工]ラー', (20, 270, 530, 815), minimum=0)
            below = [row for row in rows if row['box'][1] >= title['box'][1] + title['box'][3]]
            if any('リトライ' in row['text'] for row in below):
                raise ValueError('current communication dialog offers retry')
            close = _row(below, r'閉じる', (175, 570, 370, 675), minimum=0)
            return dict(scene='session_expired', target=tuple(close['box']))
        _row(rows, r'通信エラー', (30, 930, 300, 85))
        _row(rows, r'時間が経っているのでタイトルに戻します', (40, 1020, 560, 80))
        close = _row(rows, r'閉じる', (300, 1120, 160, 80))
        return dict(scene='session_expired', target=_inside_target(close, (355, 1150, 25, 10)))
    collect(expired)
    if nonbattle and not candidates:
        collect(title)
    if len(candidates) != 1:
        raise ValueError('saved-cultivation recovery page is unknown or contradictory')
    return candidates[0]


def verify_saved_scope(host, source, handoff_id):
    from live_decision.hif_session import HifRoundContext
    if not isinstance(source, dict):
        raise ValueError('saved-cultivation business source is not a record')
    handoff = host.handoff or {}
    owner = host.ownership or {}
    live = handoff.get('live_context')
    if live is None:
        from live_decision.hif_navigation_recovery import verify_unbound_opening_source

        if (owner.get('owner') != 'task275' or owner.get('phase') != 'battle'
                or owner.get('game_pid') != host.game_pid or handoff.get('handoff_id') != handoff_id
                or (handoff.get('idol'), handoff.get('plan')) != active_hif_role(host)
                or type(handoff.get('round_number')) is not int or handoff['round_number'] not in (1, 2)
                or handoff.get('terminal_receipt') is not None
                or not isinstance(source.get('id'), str) or not re.fullmatch(r'[0-9]+-[a-z0-9-]+', source['id'])
                or host.ledger is None or host.ledger.pending is not None
                or host.ledger.state.run_id != host.run_id or host.ledger.state.day != 7
                or host.ledger.state.revision != handoff.get('ledger_revision')):
            raise ValueError('saved-cultivation opening differs from the active handoff/business source')
        verify_unbound_opening_source(dict(run_id=host.run_id, ownership=owner, handoff=handoff,
            ledger=host.ledger.record()), source)
        return
    opening = source.get('handoff_opening', {})
    observed = source.get('observed_live_context', {})
    round_context = source.get('round_context', {})
    if any(not isinstance(value, dict) for value in (handoff, live, opening, observed, round_context)):
        raise ValueError('saved-cultivation source metadata is incomplete')
    try:
        HifRoundContext(**round_context)
    except (TypeError, ValueError) as error:
        raise ValueError('saved-cultivation round proof is incomplete') from error
    if (owner.get('owner') != 'task275' or owner.get('phase') != 'battle'
            or owner.get('game_pid') != host.game_pid or handoff.get('handoff_id') != handoff_id
            or (handoff.get('idol'), handoff.get('plan')) != active_hif_role(host)
            or type(handoff.get('round_number')) is not int or handoff['round_number'] not in (1, 2)
            or live.get('round_number') != handoff.get('round_number')
            or handoff.get('terminal_receipt') is not None
            or not isinstance(source.get('id'), str) or not re.fullmatch(r'[0-9]+-[a-z0-9-]+', source['id'])
            or source.get('endpoints', {}).get('game', {}).get('pid') != host.game_pid
            or opening.get('handoff_id') != handoff_id or opening.get('route_run_id') != host.run_id
            or opening.get('policy_profile') != handoff.get('policy_profile')
            or opening.get('ledger_revision') != handoff.get('ledger_revision')
            or (opening.get('live_run_id'), opening.get('live_revision'), opening.get('round_number'))
                != (live.get('run_id'), live.get('revision'), handoff.get('round_number'))
            or (observed.get('run_id'), observed.get('revision'), round_context.get('round_number'))
                != (live.get('run_id'), live.get('revision'), handoff.get('round_number'))
            or (round_context.get('run_id'), round_context.get('revision'))
                != (live.get('run_id'), live.get('revision'))
            or (observed.get('idol'), observed.get('plan'), observed.get('max_stamina'))
                != (handoff.get('idol'), handoff.get('plan'), handoff.get('max_stamina'))
            or host.ledger is None or host.ledger.pending is not None
            or host.ledger.state.run_id != host.run_id or host.ledger.state.day != 7
            or host.ledger.state.revision != handoff.get('ledger_revision')):
        raise ValueError('saved-cultivation recovery differs from the active handoff/business source')


ACTION = {'date_changed': 'date_return_title', 'data_update': 'date_return_title', 'game_title': 'date_enter_title',
    'login_bonus': 'date_login_bonus', 'friend_notice': 'date_friend_notice',
    'saved_home': 'date_open_saved', 'resume_saved': 'date_resume_saved', 'network_error': 'date_network_retry',
    'session_expired': 'date_return_title'}
EXPECTED = {'date_return_title': ('game_title', 'network_error'),
    'date_enter_title': ('login_bonus', 'friend_notice', 'saved_home', 'network_error'),
    'date_login_bonus': ('login_bonus', 'friend_notice', 'saved_home', 'network_error'),
    'date_friend_notice': ('saved_home', 'network_error', 'login_bonus'),
    'date_open_saved': ('resume_saved', 'network_error'),
    'date_resume_saved': ('saved_live', 'saved_terminal', 'network_error', 'session_expired')}


class DateRecoveryProducer:
    def __init__(self, context, *, run_id, game_pid, handoff_id, source_report, clock,
            evidence_sink=None, recovery_report=None, navigation_guard=None):
        if navigation_guard is not None and not callable(navigation_guard):
            raise ValueError('saved-cultivation navigation budget guard is not callable')
        self.context, self.run_id, self.game_pid = context, run_id, game_pid
        self.handoff_id, self.source = handoff_id, deepcopy(source_report)
        self.clock, self.evidence_sink, self.recovery_report = clock, evidence_sink, recovery_report
        self.host, self.rows = None, []
        self.events, self.last_frame_ids = [], ()
        self._new_episode_state = None
        self.navigation_guard = navigation_guard

    def _guard(self):
        if self.navigation_guard is not None and self.navigation_guard() is not True:
            raise ValueError('saved-cultivation navigation budget is not active')

    def _initial_state(self):
        return dict(schema_version=2, business_scope=_business_scope(self.host), revision=0, episode=1, previous_episodes=[],
            source_result=self.source['id'], handoff_id=self.handoff_id,
            resume_requested=False, retry_counts=dict(login=0, resume=0), retry_reservation=None,
            completed=False, completion_frame_ids=[], failures=[])

    def _state(self, *, allow_prior_source=False):
        if self._new_episode_state is not None:
            return deepcopy(self._new_episode_state)
        state = (self.host.handoff or {}).get('nonbattle_context', {}).get('date_recovery')
        if state is None:
            return self._initial_state()
        fields = {'schema_version', 'business_scope', 'revision', 'source_result',
                'handoff_id', 'resume_requested', 'retry_counts', 'retry_reservation', 'completed',
                'completion_frame_ids', 'failures', 'episode', 'previous_episodes'}
        if (not isinstance(state, dict) or set(state) != fields
                or type(state['schema_version']) is not int or state['schema_version'] != 2
                or state['business_scope'] != _business_scope(self.host)
                or type(state['revision']) is not int or state['revision'] < 1
                or type(state['episode']) is not int or state['episode'] < 1
                or not isinstance(state['previous_episodes'], list)
                or not isinstance(state['source_result'], str)
                or not re.fullmatch(r'[0-9]+-[a-z0-9-]+', state['source_result'])
                or (not allow_prior_source and state['source_result'] != self.source['id'])
                or state['handoff_id'] != self.handoff_id
                or type(state['resume_requested']) is not bool or type(state['completed']) is not bool
                or not isinstance(state['retry_counts'], dict) or set(state['retry_counts']) != {'login', 'resume'}
                or any(type(value) is not int or value not in (0, 1) for value in state['retry_counts'].values())
                or not isinstance(state['failures'], list) or not isinstance(state['completion_frame_ids'], list)):
            raise ValueError('saved-cultivation navigation metadata changed or belongs to another source')
        reservation = state['retry_reservation']
        pending = self.host.hooks.transition_gate.pending
        if reservation is not None and (
                not isinstance(reservation, dict) or set(reservation) != {'phase', 'revision', 'issued_at_ms'}
                or reservation['phase'] not in ('login', 'resume')
                or state['retry_counts'][reservation['phase']] != 1
                or type(reservation['revision']) is not int or type(reservation['issued_at_ms']) is not int
                or pending is None or pending.target.kind != 'date_network_retry'
                or (reservation['revision'], reservation['issued_at_ms'])
                    != (pending.source.revision, self.host.hooks.transition_gate.issued_at_ms)):
            raise ValueError('saved-cultivation retry reservation differs from its pending')
        frames = state['completion_frame_ids']
        if state['completed'] and (pending is not None or len(frames) != 1
                or not isinstance(frames[0], str) or not frames[0]):
            raise ValueError('saved-cultivation completion needs one destination frame')
        return deepcopy(state)

    def checkpoint(self, gate, *, cancel_unissued=False):
        self._guard()
        state = self._state()
        previous, pending = self.host.hooks.transition_gate.pending, gate.pending
        page = self.host._page
        if pending is not None and previous is None:
            if pending.target.kind == 'date_resume_saved':
                if state['resume_requested']:
                    raise ValueError('the original save was already requested; do not repeat it')
                state['resume_requested'] = True
            if pending.target.kind == 'date_network_retry':
                phase = 'resume' if state['resume_requested'] else 'login'
                if state['retry_counts'][phase] >= 1:
                    raise ValueError('saved-cultivation network retry budget reached')
                state['retry_counts'][phase] += 1
                state['retry_reservation'] = dict(phase=phase, revision=pending.source.revision,
                    issued_at_ms=gate.issued_at_ms)
        if previous is not None and pending is None:
            if cancel_unissued and previous.target.kind == 'date_resume_saved':
                state['resume_requested'] = False
            reservation = state['retry_reservation']
            if cancel_unissued and previous.target.kind == 'date_network_retry':
                if (not isinstance(reservation, dict) or reservation.get('revision') != previous.source.revision
                        or reservation.get('issued_at_ms') != self.host.hooks.transition_gate.issued_at_ms):
                    raise ValueError('unissued retry does not match its original reservation')
                state['retry_counts'][reservation['phase']] -= 1
            state['retry_reservation'] = None
            if (not cancel_unissued and previous.target.kind == 'date_return_title'
                    and previous.source.scene == 'session_expired' and page is not None
                    and page.observation.scene == 'game_title'):
                # The game abandoned the previous resume. Only a confirmed
                # return to title allows a fresh request for the same save.
                state['resume_requested'] = False
        if page is not None and page.observation.scene in ('saved_live', 'saved_terminal') and pending is None:
            state.update(completed=True, completion_frame_ids=list(page.frame_ids))
        state['revision'] += 1
        return state

    def _page(self, image):
        from live_decision.hif_recovery import read_recovery_page

        from .production import _current_rows
        self.rows = _current_rows(self.context, image)
        try:
            return read_date_page(image, self.rows, context=self.context)
        except ValueError as error:
            if any(row['score'] >= .95 and row['text'].startswith('開催期間:') for row in self.rows):
                from .checkpoint import restore_transition
                from .input_receipt import original_input_report
                from .login_display import claimed_login_display

                state = self._state()
                carrier = (self.host.handoff or {}).get('nonbattle_context', {}).get('route_input_receipt') or {}
                prior = restore_transition(carrier.get('original_transition'), clock_epoch=self.clock.epoch)
                original_input_report(self.host, prior)
                if (state['completed'] or state['resume_requested'] or prior.pending is None
                        or prior.pending.target.kind not in ('date_enter_title', 'date_login_bonus')
                        or not prior.pending.source.page_id.startswith(f'date:{self.source["id"]}:{self.handoff_id}:')):
                    raise ValueError('claimed login display is outside its original title navigation')
                return claimed_login_display(image, self.rows)
            blockers = {'日付変更', '日付が変わりました', '日付が変わりました。', 'タイトルへ',
                '通信エラー', '通信中にエラーが発生しました', 'リトライ',
                'プロデュース再開', 'プロデュース中のアイドル', 'LOGINBONUS',
                'ログインボーナス', 'ログインボーナズ', 'フレンドコイン獲得', '使用期限確認'}
            if any(row['score'] >= .85 and (row['text'] in blockers or re.search(r'親愛度.*解放', row['text']))
                    for row in self.rows):
                raise error
            terminal = self._saved_terminal()
            if terminal is not None:
                return dict(scene='saved_terminal', target=None, terminal_result=terminal)
            page = read_recovery_page(self.context, image, round_number=self.host.handoff['round_number'])
            if (page['phase'] != 'live' or not 0 <= page['stamina'] <= self.host.handoff['max_stamina']
                    or not 1 <= page['turns'] <= 99):
                raise error
            return dict(scene='saved_live', target=None, stamina=page['stamina'], turns=page['turns'])

    def _saved_terminal(self):
        """Recognize arrival only; Native still verifies its original pending."""
        from types import SimpleNamespace

        from live_decision.hif_result import _read_hif_result
        from live_decision.hif_session import HifRoundContext
        from live_decision.hif_profiles import supported_hif_identity

        from .checkpoint import restore_transition
        from .input_receipt import original_input_report

        if not self._state()['resume_requested']:
            return None
        carrier = (self.host.handoff.get('nonbattle_context') or {}).get('route_input_receipt') or {}
        original = restore_transition(carrier.get('original_transition'), clock_epoch=self.clock.epoch)
        if original.pending is None or original.pending.target.kind != 'date_resume_saved':
            return None
        if not original.pending.source.page_id.startswith(f'date:{self.source["id"]}:{self.handoff_id}:'):
            raise ValueError('saved terminal lacks its original same-save resume')
        original_input_report(self.host, original)
        profile = supported_hif_identity(self.source['observed_live_context']['idol'],
            self.source['observed_live_context']['plan'])
        return _read_hif_result([SimpleNamespace(**row) for row in self.rows], idol=profile.idol,
            round_context=HifRoundContext(**self.source['round_context']))

    def observe(self, *, reconciliation=False):
        verify_saved_scope(self.host, self.source, self.handoff_id)
        deadline = self.clock.now() + 8000
        gate = self.host.hooks.transition_gate
        if gate.pending is not None and not reconciliation:
            deadline = min(deadline, gate.issued_at_ms + gate.timeout_ms)
        while True:
            self._guard()
            started = self.clock.now()
            if started >= deadline or self.context.tasker.stopping:
                raise ValueError('saved-cultivation capture deadline reached')
            job = self.context.tasker.controller.post_screencap().wait()
            if not job.succeeded:
                raise ValueError('saved-cultivation Maa capture failed')
            image = job.get()
            key = f'date-frame-{self.game_pid}-{started}'
            try:
                page = self._page(image)
            except ValueError as error:
                event = dict(frame_id=key, phase='unknown', reason=str(error), rows=self.rows)
                self.events.append(event)
                if self.evidence_sink:
                    self.evidence_sink(image, event)
                if gate.stopped or self.clock.now() >= deadline:
                    raise
                time.sleep(.1)
                continue
            if self.evidence_sink:
                self.evidence_sink(image, dict(frame_id=key, phase=page['scene'], observation=page))
            break
        revision = self.clock.now()
        kind = ACTION.get(page['scene'])
        fields = ':'.join(str(page.get(name, '')) for name in ('turns', 'stamina'))
        if page.get('page_kind'):
            fields += ':'+page['page_kind']
        if page.get('terminal_result'):
            fields += ':'+json.dumps(page['terminal_result'], sort_keys=True)
        observation = Observation(self.run_id, revision, page['scene'],
            f'date:{self.source["id"]}:{self.handoff_id}:{page["scene"]}:{fields}',
            7, 'unverified', 'unverified', revision - started, 1, 'HIF',
            (Target(kind, kind, available=True),) if kind else ())
        self.events.append(dict(phase=page['scene'], frame_ids=[key]))
        self.last_frame_ids = (key,)
        return HifRoutePage(observation, {kind: tuple(page['target'])} if kind else {}, frame_ids=(key,))

    def decide(self, observation, config):
        self._guard()
        error = validate_observation(observation)
        if error:
            return Decision(Outcome.STOP, error)
        if observation.scene in ('saved_live', 'saved_terminal'):
            state = self._state(allow_prior_source=True)
            if not state['completed']:
                self.host._save(transition_checkpoint(self.host.hooks.transition_gate, self.clock.epoch))
            return Decision(Outcome.PAUSE, f'{observation.scene}_requires_original_business_reconciliation')
        state = self._state()
        if state['completed']:
            return Decision(Outcome.STOP, 'saved_cultivation_navigation_already_completed')
        kind = ACTION.get(observation.scene)
        if kind not in SAVED_CULTIVATION_NAVIGATION:
            return Decision(Outcome.STOP, 'unknown_saved_cultivation_navigation')
        if kind == 'date_resume_saved' and state['resume_requested']:
            return Decision(Outcome.STOP, 'saved_cultivation_resume_already_requested')
        if kind == 'date_network_retry':
            phase = 'resume' if state['resume_requested'] else 'login'
            if state['retry_counts'][phase] >= 1:
                return Decision(Outcome.STOP, 'saved_cultivation_network_retry_exhausted')
            expected = ('saved_live',) if state['resume_requested'] else ('login_bonus', 'friend_notice', 'saved_home', 'resume_saved')
            expected += ('session_expired',)
        else:
            expected = EXPECTED[kind]
        return propose_unique(observation, observation.targets, f'hif-existing-save:{kind}', expected)

    def prepare(self, gate):
        verify_saved_scope(self.host, self.source, self.handoff_id)
        state = self._state(allow_prior_source=True)
        if state['source_result'] != self.source['id'] or state['completed']:
            if not state['completed'] or gate.pending is not None or gate.stopped:
                raise ValueError('another saved-cultivation navigation is still unresolved')
            proof = self.observe(reconciliation=True)
            self.host._page = proof
            if proof.observation.scene in ('saved_live', 'saved_terminal'):
                return gate
            if (proof.observation.scene != 'date_changed'
                    or not 0 <= proof.observation.capture_age_ms + self.clock.now() - proof.observation.revision <= self.host.hooks.max_input_age_ms
                    or set(proof.frame_ids) & set(state['completion_frame_ids'])):
                raise ValueError('another recovery episode requires a fresh complete date-change dialog')
            previous = deepcopy(state)
            previous.pop('previous_episodes')
            next_state = self._initial_state()
            next_state.update(revision=state['revision'], episode=state['episode'] + 1,
                previous_episodes=[*state['previous_episodes'], previous])
            self._new_episode_state = next_state
            try:
                self.host._save(transition_checkpoint(gate, self.clock.epoch))
            finally:
                self._new_episode_state = None
            return gate
        self._state()
        if gate.pending is not None:
            if gate.pending.target.kind not in SAVED_CULTIVATION_NAVIGATION:
                raise ValueError('saved-cultivation navigation cannot clear another pending action')
            carrier = (self.host.handoff or {}).get('nonbattle_context', {}).get('route_input_receipt') or {}
            if carrier.get('dispatch_state') == 'cancelled_unissued':
                return self._cancel_rejected(gate)
            if self.recovery_report is not None:
                return self._reconcile(gate)
            from .input_receipt import original_input_report
            original = original_input_report(self.host, gate)
            return self._reconcile(gate, original_report=original)
        self.host._save(transition_checkpoint(gate, self.clock.epoch))
        return gate

    def _cancel_rejected(self, gate):
        from .input_receipt import cancelled_unissued_report

        report = cancelled_unissued_report(self.host, gate)
        event = report['route']['input_events'][0]
        proof = self.observe(reconciliation=True)
        seen, pending = proof.observation, gate.pending
        if (seen.page_id != pending.source.page_id or seen.scene != pending.source.scene
                or seen.revision <= pending.source.revision or validate_observation(seen) is not None
                or not source_frames_match(seen, proof.frame_ids)
                or set(proof.frame_ids) & {row.get('frame_id') for row in report.get('frames', ())}
                or not 0 <= seen.capture_age_ms+self.clock.now()-seen.revision <= self.host.hooks.max_input_age_ms
                or list(proof.boxes.get(pending.target.key, ())) != event.get('target')
                or not pending.source.page_id.startswith(f'date:{self.source["id"]}:{self.handoff_id}:')):
            raise ValueError('unissued date cancellation lacks its fresh original source and target')
        current = json.loads(self.host.path.read_text(encoding='utf-8'))
        if current != self.host._prepared_business_record():
            raise ValueError('unissued date cancellation cannot overwrite a changed business record')
        state = self._state()
        state['failures'].append(dict(outcome='verified_unissued_date_input_cancelled',
            original_transition=transition_checkpoint(gate, self.clock.epoch),
            rejection_reason=event['rejection_reason'], original_report=report,
            receipt_frame_ids=list(proof.frame_ids)))
        previous = self.host.handoff
        handoff, context = dict(previous), dict(previous.get('nonbattle_context') or {})
        context['date_recovery'] = state
        handoff['nonbattle_context'] = context
        self.host.handoff = handoff
        try:
            self.host._cancel_unissued()
        except BaseException:
            self.host.handoff = previous
            raise
        return self.host.hooks.transition_gate

    def _reconcile(self, gate, *, original_report=None):
        report, pending = original_report if original_report is not None else self.recovery_report, gate.pending
        automatic = original_report is not None
        if automatic and not pending.source.page_id.startswith(
                f'date:{self.source["id"]}:{self.handoff_id}:{pending.source.scene}:'):
            raise ValueError('original date dispatch lacks its bound business source and handoff')
        route = report.get('route' if automatic else 'date_recovery', {})
        matching = [row for row in route.get('input_events', []) if row.get('target_kind') == pending.target.kind
            and row.get('target_key') == pending.target.key and row.get('source_revision') == pending.source.revision
            and row.get('issued_at_ms') == gate.issued_at_ms and row.get('dispatch_attempted') is True
            and row.get('success') is True and row.get('uncertain') is False]
        if (route.get('run_id') != self.run_id or not automatic and route.get('business_source') != self.source['id']
                or report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid or len(matching) != 1):
            raise ValueError('saved-cultivation recovery report differs from the original navigation')
        proof = self.observe(reconciliation=True)
        if (proof.observation.scene not in pending.expected_scenes
                or proof.observation.state_key == pending.source.state_key
                or proof.observation.revision <= pending.source.revision
                or not 0 <= proof.observation.capture_age_ms + self.clock.now() - proof.observation.revision <= self.host.hooks.max_input_age_ms
                or not source_frames_match(proof.observation, proof.frame_ids)
                or set(proof.frame_ids) & {row.get('frame_id') for row in report.get('frames', [])}):
            raise ValueError('saved-cultivation late destination is not independently confirmed')
        state = self._state()
        state['failures'].append(dict(source_result=report['id'],
            original_transition=transition_checkpoint(gate, self.clock.epoch),
            receipt_frame_ids=list(proof.frame_ids), destination=proof.observation.scene))
        previous = self.host.handoff
        self.host.handoff = deepcopy(previous)
        self.host.handoff.setdefault('nonbattle_context', {})['date_recovery'] = state
        self.host._page = proof
        settled = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            self.host._save(transition_checkpoint(settled, self.clock.epoch))
        except BaseException:
            self.host.handoff = previous
            raise
        return settled


def verify_completed_date_return(shared_record, source_report):
    """Read an existing completed receipt; never grant input or settle battle.

    The recorded destination identifies the completed navigation. The battle
    owner must still capture fresh business evidence and reconcile its original
    native transaction before input.
    """
    import csv
    from types import SimpleNamespace

    from live_decision.catalog import SkillCardFeatureCatalog

    from .ledger import restore_ledger
    from .ownership import validate_ownership
    from .checkpoint import restore_transition

    record = deepcopy(shared_record)
    if (not isinstance(record, dict) or record.get('schema_version') != 3
            or not isinstance(record.get('handoff'), dict)):
        raise ValueError('completed date return lacks its shared record')
    owner = validate_ownership(record.get('ownership'))
    raw_gate = record.get('transition') or {}
    gate = restore_transition(raw_gate, clock_epoch=raw_gate.get('clock_epoch'))
    if gate.pending is not None or gate.stopped:
        raise ValueError('completed date return still has an unresolved navigation')
    data = resolve_runtime_path(Path('assets/data'), Path('data'))
    with (data / 'p_drinks.csv').open(encoding='utf-8-sig', newline='') as stream:
        drinks = {row['id'] for row in csv.DictReader(stream)}
    ledger = restore_ledger(record.get('ledger'),
        card_catalog=SkillCardFeatureCatalog.load(data / 'skill_cards.csv'), drink_ids=drinks)
    handoff = record['handoff']
    host = SimpleNamespace(run_id=record.get('run_id'), game_pid=owner['game_pid'], ownership=owner,
        ledger=ledger, handoff=handoff, hooks=SimpleNamespace(transition_gate=gate))
    verify_saved_scope(host, source_report, handoff.get('handoff_id'))
    producer = DateRecoveryProducer(None, run_id=host.run_id, game_pid=host.game_pid,
        handoff_id=handoff['handoff_id'], source_report=source_report, clock=None)
    producer.host = host
    state = producer._state()
    if not state['completed'] or state['retry_reservation'] is not None:
        raise ValueError('completed date return lacks a frozen original business scope and completion')
    return dict(source_result=state['source_result'], handoff_id=state['handoff_id'],
        episode=state['episode'], revision=state['revision'], completion_frame_ids=list(state['completion_frame_ids']))


def bind_date_recovery_runtime(context, *, path, run_id, handoff_id, game_pid, client_size,
        source_report, authorise_capture, authorise_input, evidence_sink=None, recovery_report=None,
        navigation_guard=None, authorise_preparation=None):
    """Bind on a fresh Maa context, outside any active HifInputSession.

    The caller supplies the same separate permissions as the route host. The
    source report supplies old business identity only, never a navigation grant.
    """
    import csv

    from utils.maa_input import verify_maa_input_endpoints
    from live_decision.catalog import SkillCardFeatureCatalog
    from live_decision.hif_capture_adapter import adapt_hif_context

    from .host import windows_route_clock
    if not Path(path).is_file():
        raise FileNotFoundError('saved-cultivation recovery cannot initialize a new run')
    if (not callable(authorise_capture) or not callable(authorise_input) or authorise_capture() is not True):
        raise ValueError('saved-cultivation capture permission is not bound')
    if authorise_preparation is not None and not callable(authorise_preparation):
        raise ValueError('saved-cultivation preparation permission is not callable')
    context = adapt_hif_context(context)
    if context.tasker.controller.set_screenshot_target_long_side(1280) is not True:
        raise ValueError('saved-cultivation capture profile was not applied')
    size = tuple(client_size)
    context.gkh_hif_client_size = size
    def endpoint():
        verified = verify_maa_input_endpoints(context.tasker.controller.info, expected_client_size=size)
        if verified['game']['pid'] != game_pid:
            raise ValueError('saved-cultivation game process changed')
        return verified
    endpoint()
    clock = windows_route_clock()
    producer = DateRecoveryProducer(context, run_id=run_id, game_pid=game_pid, handoff_id=handoff_id,
        source_report=source_report, clock=clock, evidence_sink=evidence_sink, recovery_report=recovery_report,
        navigation_guard=navigation_guard)
    data = resolve_runtime_path(Path('assets/data'), Path('data'))
    catalog = SkillCardFeatureCatalog.load(data / 'skill_cards.csv')
    with (data / 'p_drinks.csv').open(encoding='utf-8-sig', newline='') as stream:
        drinks = {row['id'] for row in csv.DictReader(stream)}
    host = HifRouteHost(path, run_id=run_id, observe=producer.observe,
        authorise_capture=authorise_capture,
        authorise_input=(authorise_preparation if authorise_preparation is not None
            else lambda: authorise_input() is True and bool(endpoint())),
        clock=clock, card_catalog=catalog, drink_ids=drinks, input_owner='task275', game_pid=game_pid,
        recovery_decide=producer.decide, recovery_prepare=producer.prepare,
        navigation_only=True, navigation_checkpoint=producer.checkpoint)
    producer.host = host
    verify_saved_scope(host, producer.source, handoff_id)
    # A previous Live input guard cannot be replaced. A stale/reused context
    # fails here; the normal entry receives a fresh original Maa context.
    context = adapt_hif_context(context, input_guard=lambda: host._active
        and host.ownership['owner'] == 'task275' and host.ownership['phase'] == 'battle'
        and (navigation_guard is None or navigation_guard() is True) and bool(endpoint()))
    producer.context = context
    host._input_adapter = context.gkh_hif_capture_adapter
    context.gkh_hif_date_recovery_host, context.gkh_hif_date_recovery_producer = host, producer
    # The product Context validates the actual current host's exact intent at
    # dispatch. Its previous nonbattle host cannot validate date navigation.
    context.gkh_hif_route_host = host
    return context


def run_date_recovery(context):
    host = context.gkh_hif_date_recovery_host
    return host.run(context, HifConfig(mode='advance'), max_actions=None, timeout_ms=None)
