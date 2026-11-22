"""Persistent HIF route host, bound explicitly by the authorised Maa client.

No capture or input grant is inferred from configuration or a saved record.
A missing checkpoint must be deliberately initialized for a newly verified run.
"""
from __future__ import annotations

import os
import re
import json
import ctypes
from copy import deepcopy
from typing import Callable
from pathlib import Path
from dataclasses import replace, dataclass

from .hif import active_hif_role, decide_hif
from .engine import (
    Outcome,
    Decision,
    Observation,
    TransitionGate,
    ledger_frames_match,
    source_frames_match,
    validate_observation,
    current_completion_next,
)
from .ledger import HifLedger, LedgerEffect, HifLedgerState, restore_ledger
from .deletion import DeletionMenu
from .ownership import write_hif_record, locked_hif_record, validate_ownership
from .checkpoint import restore_transition, transition_checkpoint
from .maa_bridge import MAX_INPUT_AGE_MS, MAX_NAVIGATION_INPUT_AGE_MS, HifRuntimeHooks, run_hif_sequence
from .deck_history import record_ledger_deck
from .customization import CustomizationMenu, CustomizationResult, plan_customization
from .drink_replacement import DrinkReplacementMenu

RECOVERY_NAVIGATION = frozenset({'recovery_menu', 'recovery_notebook',
    'recovery_identity', 'recovery_close', 'recovery_inventory',
    'recovery_inventory_detail', 'recovery_inventory_scroll', 'end_result_next', 'end_defeat_exit',
    'end_star_reward_next', 'end_final_evaluation_next', 'end_generate_memory', 'end_memory_complete_next',
    'end_memory_detail_next', 'end_produce_complete_next', 'round_one_result_next', 'round_one_award_next',
    'recovery_drink_detail', 'recovery_drink_close'})
SAVED_CULTIVATION_NAVIGATION = frozenset({'date_return_title', 'date_enter_title',
    'date_login_bonus', 'date_friend_notice', 'date_open_saved', 'date_resume_saved', 'date_network_retry'})


@dataclass(frozen=True)
class RouteClock:
    epoch: str
    now: Callable[[], int]


def windows_route_clock() -> RouteClock:
    """Use boot time and uptime from Windows, never estimated wall-clock time."""
    if os.name != 'nt':
        raise RuntimeError('HIF persistent host needs a verified Windows clock')

    class TimeOfDay(ctypes.Structure):
        _fields_ = [('boot', ctypes.c_int64), ('current', ctypes.c_int64),
                    ('bias', ctypes.c_int64), ('zone', ctypes.c_uint32),
                    ('reserved', ctypes.c_uint32), ('boot_bias', ctypes.c_uint64),
                    ('sleep_bias', ctypes.c_uint64)]

    ntdll = ctypes.WinDLL('ntdll')
    query = ntdll.NtQuerySystemInformation
    query.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    query.restype = ctypes.c_long
    info = TimeOfDay()
    if query(3, ctypes.byref(info), ctypes.sizeof(info), None) != 0 or info.boot <= 0:
        raise RuntimeError('HIF Windows boot epoch unavailable')
    kernel = ctypes.WinDLL('kernel32')
    kernel.GetTickCount64.argtypes = []
    kernel.GetTickCount64.restype = ctypes.c_uint64
    return RouteClock(f'windows-boot:{info.boot}', lambda: int(kernel.GetTickCount64()))


@dataclass(frozen=True)
class HifRoutePage:
    observation: Observation
    boxes: dict
    baseline: HifLedgerState | None = None
    frame_ids: tuple[str, ...] = ()
    effects: tuple[LedgerEffect, ...] = ()
    receipt: HifLedgerState | None = None
    receipt_frame_ids: tuple[str, ...] = ()
    customization: CustomizationMenu | None = None
    customization_result: CustomizationResult | None = None
    deletion: DeletionMenu | None = None
    drink_replacement: DrinkReplacementMenu | None = None
    closing: dict | None = None
    start: dict | None = None
    reward: dict | None = None
    activity: dict | None = None
    drink_inventory: dict | None = None
    customization_receipt: dict | None = None
    schedule_countdown_frame_id: str | None = None



class HifRouteHost:
    """Own one run's durable transition and reusable hooks across Maa calls."""

    def __init__(self, path: str | Path, *, run_id: str, observe,
                 authorise_capture, authorise_input, clock: RouteClock | None = None,
                 new_verified_run: bool = False, ledger: HifLedger | None = None,
                 card_catalog=None, drink_ids=(), customization_catalog=None, customization_menu_source=None,
                 ownership=None, input_owner=None, game_pid=None, recovery_decide=None,
                 recovery_prepare=None, nonbattle_decide=None, navigation_only=False,
                 navigation_checkpoint=None, ending_terminal_report=None):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError('HIF host requires a verified run identity')
        self.path = Path(path)
        self.run_id = run_id
        self._active = False
        self.input_events = []
        self._input_source = None
        self._input_adapter = None
        self.ownership = validate_ownership(ownership) if ownership is not None else None
        self.handoff = None
        self._ending_terminal_report = ending_terminal_report
        self.input_owner = input_owner
        self.game_pid = game_pid
        self._recovery_decide = recovery_decide
        self._recovery_prepare = recovery_prepare
        self._nonbattle_decide = nonbattle_decide
        if (type(navigation_only) is not bool or navigation_only and (
                new_verified_run or input_owner != 'task275' or not callable(recovery_decide)
                or not callable(navigation_checkpoint))):
            raise ValueError('saved-cultivation navigation needs the existing battle owner and bound producer')
        self._navigation_only, self._navigation_checkpoint = navigation_only, navigation_checkpoint
        # Known nonconsuming Date/title/login/save navigation has its own
        # bounded window. Business inputs retain their existing one-second gate.
        self._max_input_age_ms = MAX_NAVIGATION_INPUT_AGE_MS if navigation_only else MAX_INPUT_AGE_MS
        self.ledger = ledger
        self._catalog = card_catalog if card_catalog is not None else (ledger.card_catalog if ledger else None)
        self._drink_ids = frozenset(drink_ids) if ledger is None else ledger.drink_ids
        self._page = None
        self._confirmed_ledger = None
        self._confirmed_reward = None
        self._confirmed_activity = None
        self._confirmed_customization = None
        self._confirmed_recovery_drinks = None
        self._customization_catalog = customization_catalog
        if customization_menu_source is not None and not callable(customization_menu_source):
            raise ValueError('customization menu source must be a trusted producer')
        self._customization_menu_source = customization_menu_source
        if ledger is not None and (ledger.state.run_id != run_id or ledger.pending is not None):
            raise ValueError('HIF new ledger must belong to the verified run without pending')
        self.clock = clock or windows_route_clock()
        if not isinstance(self.clock.epoch, str) or not self.clock.epoch:
            raise ValueError('HIF host clock epoch missing')
        now = self.clock.now()
        if type(now) is not int or now < 0:
            raise ValueError('HIF host clock invalid')
        # Never overwrite a pending/stopped or corrupt file in "new run" mode.
        if self.path.exists():
            if new_verified_run:
                raise ValueError('HIF checkpoint already exists; use a distinct new run path')
            gate = self._load()
        elif new_verified_run:
            gate = TransitionGate()
            self._save(transition_checkpoint(gate, self.clock.epoch), create=True)
        else:
            raise FileNotFoundError('HIF checkpoint missing; no automatic cold start')
        if gate.pending is not None and gate.pending.source.run_id != run_id:
            raise ValueError('HIF pending checkpoint belongs to another run')

        def bound_observe():
            raw = observe()
            if isinstance(raw, HifRoutePage) and any(key.startswith('route-frame-') for key in raw.frame_ids):
                matches = [re.fullmatch(r'route-frame-' + str(self.game_pid) + r'-([0-9]+)', key)
                    for key in raw.frame_ids]
                if not source_frames_match(raw.observation, raw.frame_ids) or not all(matches):
                    raise ValueError('current route observation lacks its distinct same-game source frames')
                captures = [int(match[1]) for match in matches]
                if captures != sorted(captures) or captures[-1] > raw.observation.revision:
                    raise ValueError('current route source frames do not precede their original observation')
                raw = replace(raw, observation=replace(raw.observation,
                    capture_age_ms=max(raw.observation.capture_age_ms, raw.observation.revision - captures[-1])))
            self._page = raw if isinstance(raw, HifRoutePage) else None
            observation, boxes = (raw.observation, raw.boxes) if self._page is not None else raw
            if observation.run_id != self.run_id:
                raise ValueError('HIF observation belongs to another run')
            return observation, boxes

        self.hooks = HifRuntimeHooks(bound_observe, authorise_capture,
            lambda: self._active and authorise_input() is True,
            transition_gate=gate, save_transition=self._save,
            clock_epoch=self.clock.epoch, clock=self.clock.now,
            confirmation_ready=self._confirmation_ready, cancel_unissued=self._cancel_unissued,
            decide=self._decide, input_event=self._record_input_event, input_result=self._record_input_result,
            confirmation_transient=self._confirmation_transient,
            proposal_timeout_ms=self._proposal_timeout_ms,
            schedule_home_ready=self._schedule_home_ready,
            max_input_age_ms=self._max_input_age_ms if navigation_only else self._input_age_limit)

    def _schedule_home_ready(self, observation):
        from .calendar import FAMILIES
        page = self._page
        return (page is not None and page.observation == observation
            and FAMILIES.get(observation.day) == observation.scene
            and page.schedule_countdown_frame_id is not None
            and page.schedule_countdown_frame_id not in page.frame_ids)

    def _input_age_limit(self, proposal):
        if (self._navigation_only or self.ownership is not None and self.ownership['phase'] == 'recovery'
                and self.ledger is not None and self.ledger.pending is None
                and proposal.target.kind in RECOVERY_NAVIGATION):
            return MAX_NAVIGATION_INPUT_AGE_MS
        if (self.ownership is not None and self.ownership['owner'] == 'task600'
                and self.ownership['phase'] == 'nonbattle' and self.ledger is not None):
            from .supply import NAVIGATION_DESTINATIONS, navigation_effect
            if proposal.target.kind in NAVIGATION_DESTINATIONS and proposal.target.key == proposal.target.kind:
                pending = self.ledger.pending
                effects = ((pending.effect,) if pending is not None else
                    tuple(effect for effect in (getattr(self, '_page', None).effects if getattr(self, '_page', None) else ())
                        if effect.target_key == proposal.target.key))
                if effects == (navigation_effect(self.ledger.state, proposal.target.key),):
                    return MAX_NAVIGATION_INPUT_AGE_MS
        return MAX_INPUT_AGE_MS

    def _record_input_event(self, event):
        from .input_receipt import persist_input_event
        # Keep the same mutable audit item for existing dispatch accounting.
        self.input_events.append(event)
        if self.ownership is not None and (self._page is not None or self._input_source is not None):
            persist_input_event(self, event)

    def _record_input_result(self, event):
        if self.ownership is not None and (self._page is not None or self._input_source is not None):
            from .input_receipt import persist_input_event
            if ((event.get('rejected_before_dispatch') is True
                    or event.get('remaining_clicks_rejected_before_dispatch') is True)
                    and json.loads(self.path.read_text(encoding='utf-8')) != self._prepared_business_record()):
                raise ValueError('unissued rejection cannot overwrite a changed business record')
            persist_input_event(self, event, complete=True)

    def _date_history(self):
        handoff = dict(self.handoff)
        history = dict(handoff.get('nonbattle_context') or {})
        for key in ('date_recovery', 'route_input_receipt'):
            history.pop(key, None)
        handoff['nonbattle_context'] = history
        return handoff

    def _prime_date_history(self):
        raw = json.dumps(self._date_history(), allow_nan=False)
        self._date_history_snapshot = (raw, json.loads(raw))

    def _prepared_business_record(self):
        value = dict(schema_version=3, run_id=self.run_id,
            transition=transition_checkpoint(self.hooks.transition_gate, self.clock.epoch),
            ledger=self.ledger.record() if self.ledger is not None else None,
            ownership=self.ownership, handoff=self.handoff)
        if self._navigation_only and (self.ownership or {}).get('owner') == 'task275':
            # Format unchanged JSON history before the first source capture.
            # Every check still compares all current values and returns a fresh
            # parse, so neither a host mutation nor a caller alias is hidden.
            if not hasattr(self, '_date_history_snapshot'):
                self._prime_date_history()
            raw, original = self._date_history_snapshot
            if self._date_history() != original:
                raise ValueError('date dispatch unchanged business history was modified')
            for key in ('transition', 'ledger'):
                value[key] = json.loads(json.dumps(value[key], allow_nan=False))
            value['ownership'] = dict(self.ownership)
            handoff = json.loads(raw)
            history = handoff['nonbattle_context']
            for key in ('date_recovery', 'route_input_receipt'):
                current = self.handoff.get('nonbattle_context') or {}
                if key in current:
                    history[key] = json.loads(json.dumps(current[key], allow_nan=False))
            value['handoff'] = handoff
            return value
        return json.loads(json.dumps(value, allow_nan=False))

    def _matches_prepared_business_record(self, current):
        # Closing changes only these small branches. Compare the unchanged
        # JSON history directly, without serializing another complete copy.
        if (self.ownership is not None and self.ownership['owner'] == 'task600'
                and self.ownership['phase'] == 'recovery' and self.ledger is not None
                and self.ledger.pending is None and self._page is not None
                and isinstance(self._page.closing, dict)):
            handoff = dict(self.handoff or {})
            context = dict(handoff.get('nonbattle_context') or {})
            if 'route_input_receipt' in context:
                context['route_input_receipt'] = json.loads(json.dumps(context['route_input_receipt'], allow_nan=False))
            handoff['nonbattle_context'] = context
            for key in ('closing', 'recovery_navigation_receipts'):
                if key in handoff:
                    handoff[key] = json.loads(json.dumps(handoff[key], allow_nan=False))
            expected = dict(schema_version=3, run_id=self.run_id, ownership=self.ownership, handoff=handoff,
                transition=json.loads(json.dumps(transition_checkpoint(self.hooks.transition_gate, self.clock.epoch), allow_nan=False)),
                ledger=json.loads(json.dumps(self.ledger.record(), allow_nan=False)))
            if current == expected:
                return True
        return current == self._prepared_business_record()

    def validate_dispatch(self, args, kwargs):
        """Final check after Maa coordinate mapping; never renew a source."""
        from .maa_bridge import route_input_action
        from .input_receipt import _transition_identity

        page, gate = self._page, self.hooks.transition_gate
        if (not self._active or page is None or gate.pending is None or gate.stopped
                or page.observation != gate.pending.source or not self.input_events or self._input_adapter is None):
            raise ValueError('route dispatch lacks its active current prepared source')
        event = self.input_events[-1]
        if (event.get('dispatch_attempted') is not True or event.get('target_key') != gate.pending.target.key
                or event.get('source_revision') != page.observation.revision
                or event.get('issued_at_ms') != gate.issued_at_ms):
            raise ValueError('route dispatch intent changed after preparation')
        action = route_input_action(gate.pending.target.kind, page.boxes[gate.pending.target.key],
            day=gate.pending.source.day)
        if event.get('action') != action:
            raise ValueError('route dispatch action changed after preparation')
        if action['action'] == 'Click':
            action['target'] = self._input_adapter._rect(action['target'])
        else:
            action['begin'] = self._input_adapter._rect(action['begin'])
            action['end'] = self._input_adapter._rect(action['end'])
        if args != ('ProduceHifAdvanceClick',) or kwargs != {'pipeline_override': {'ProduceHifAdvanceClick': action}}:
            raise ValueError('route dispatch mapped action differs from its prepared intent')
        current = self._dispatch_record()
        if not self._matches_prepared_business_record(current):
            raise ValueError('route dispatch complete prepared business record changed')
        carrier = (self.handoff or {}).get('nonbattle_context', {}).get('route_input_receipt', {})
        if (carrier.get('dispatch_state') != 'prepared'
                or _transition_identity(carrier.get('original_transition')) != _transition_identity(current['transition'])
                or [row.get('frame_id') for row in carrier.get('report', {}).get('frames', ())] != list(page.frame_ids)):
            raise ValueError('route dispatch lacks its exact durable prepared receipt')
        prefix = ('date-frame-' if self._navigation_only
            and gate.pending.target.kind in SAVED_CULTIVATION_NAVIGATION else 'route-frame-')
        captures = [re.fullmatch(prefix + str(self.game_pid) + r'-([0-9]+)', key) for key in page.frame_ids]
        if not source_frames_match(page.observation, page.frame_ids) or not all(captures):
            raise ValueError('route dispatch lacks its original same-game frames')
        times = [int(match[1]) for match in captures]
        max_input_age_ms = self._input_age_limit(gate.pending)
        if times != sorted(times) or times[-1] > page.observation.revision:
            raise ValueError('route dispatch stable source frame ordering is invalid')
        # The first tap authorises this single three-tap lesson action on a
        # fresh frame. The next two taps still pass the same live owner,
        # endpoint, mapped-point and transaction checks above.
        succeeded = event.get('clicks_succeeded')
        continuing_lesson = (gate.pending.source.day == 1 and gate.pending.source.scene == 'lesson'
            and gate.pending.target.kind == 'lesson' and event.get('clicks_planned') == 3
            and succeeded in (1, 2) and event.get('clicks_attempted') == succeeded + 1)
        age = self.clock.now() - times[-1]
        if age < 0 or (age > max_input_age_ms and not continuing_lesson):
            raise ValueError('route dispatch latest source frame exceeded its input window')
        return True

    def _dispatch_record(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def _proposal_timeout_ms(self, proposal):
        if proposal.target.kind == 'start_hif_once':
            from .start_transaction import available_start
            page = self._page
            if page is None or page.observation != proposal.source or not available_start(self, page.start):
                raise ValueError('cultivation loading wait lacks its original verified start')
            return 90_000
        if proposal.target.kind in ('confirm_interval_exit', 'battle_intro_next'):
            page = self._page
            scene = 'interval_end_confirm' if proposal.target.kind == 'confirm_interval_exit' else 'battle_intro'
            if (self.ownership is None or self.ownership['phase'] != 'nonbattle' or page is None
                    or page.observation != proposal.source or proposal.source.scene != scene
                    or not source_frames_match(page.observation, page.frame_ids)):
                raise ValueError('battle loading wait lacks its complete current entry page')
            return 90_000
        if proposal.target.kind in ('round_one_result_next', 'round_one_award_next'):
            page = self._page
            scene = 'round_one_result' if proposal.target.kind == 'round_one_result_next' else 'round_one_award'
            if (self.ownership is None or self.ownership['phase'] != 'recovery' or page is None
                    or page.observation != proposal.source or (page.closing or {}).get('phase') != scene
                    or not source_frames_match(page.observation, page.frame_ids)):
                raise ValueError('round-one return lacks its complete accepted current result or award')
            return 90_000
        if proposal.target.kind in ('select_consult','confirm_outing', 'execute_outing', 'confirm_consult',
                'activity_ack_drink', 'activity_ack_card', 'activity_ack_free_drink', 'finish_consult'):
            page = self._page
            if (page is None or page.observation != proposal.source or page.activity is None
                    or not source_frames_match(page.observation, page.frame_ids)
                    or page.observation.day not in (3, 6)):
                raise ValueError('activity wait lacks its complete current bound source')
            return 30_000 if proposal.target.kind == 'select_consult' else 90_000
        if proposal.target.kind == 'training_reward_ack':
            from .rewards import request, validate_resources
            value, page = request(self), self._page
            if (value is None or value['stage'] != 'awaiting_ack' or page is None
                    or page.observation != proposal.source or page.observation.scene != 'training_result'
                    or not source_frames_match(page.observation, page.frame_ids)
                    or not isinstance(page.reward, dict) or page.reward.get('kind') != 'result'
                    or page.reward.get('attribute') != value['attribute'] or page.reward.get('special') is not value['special']):
                raise ValueError('long reward transition wait lacks its original complete course result')
            validate_resources(self, page.reward)
            return 90_000
        if proposal.target.kind != 'confirm_training':
            return 15_000
        from .training import training_effect
        page = self._page
        if (page is None or page.observation != proposal.source or page.baseline != self.ledger.state
                or page.observation.scene != 'training_selected'
                or not source_frames_match(page.observation, page.frame_ids)
                or tuple(effect for effect in page.effects if effect.target_key == 'confirm_training') != (training_effect(self),)):
            raise ValueError('long public-course wait lacks its complete original execution contract')
        # Fixed upstream TrainAuto guards the same submitted course for at
        # most 90s. 179 proves the course performance still lasts at 14.953s.
        # Bind the longer limit before input; never renew an old submission.
        return 90_000

    def _confirmation_transient(self, observation):
        pending = self.hooks.transition_gate.pending
        if (pending is not None and self._page is not None and self._page.observation == observation
                and observation.scene in ('ending_summary', 'round_one_award')):
            from .ending import same_closing_page
            if same_closing_page(pending.source, observation, self._page.closing or {}):
                return True
        if (pending is not None and pending.target.kind in ('select_consult', 'confirm_consult') and self._page is not None
                and self._page.observation == observation and observation.scene in ('consult_entry','consult_selected')):
            from .activity_reward import request
            value = request(self)
            proof = self._page.activity or {}
            return (value is not None and value.get('source_target','confirm_consult') == pending.target.kind
                and value['stage'] == 'paid_pending'
                and observation.day == value['day'] and tuple(proof.get(key) for key in ('stamina','max_stamina','currency')) == (
                    self.ledger.pending.before.stamina,self.ledger.pending.before.max_stamina,self.ledger.pending.before.currency))
        if pending is not None and pending.target.kind in ('activity_ack_support_card', 'activity_ack_support_ok'):
            from .activity_reward import request
            value = request(self)
            proof = self._page.activity if self._page is not None else None
            if value is not None and proof is not None and value['kind'] in ('support', 'consult', 'outing'):
                same = (proof.get('kind') == 'support_reward' and proof.get('grant') == value.get('current_grant')
                    if pending.target.kind == 'activity_ack_support_card' else proof.get('kind') == 'support_ok')
                if same:
                    return True
        if (pending is not None and pending.target.kind == 'activity_ack_free_drink'
                and self._page is not None and self._page.observation == observation
                and (self._page.activity or {}).get('kind') == 'consult_reward_transition'):
            self.bind_observed_activity_transition(self._page)
            return True
        if pending is not None and pending.target.kind == 'confirm_training' and observation.scene == 'training_animation':
            from .rewards import same_course_animation
            return (self._page is not None and self._page.observation == observation
                and same_course_animation(self, pending, self._page))
        if (self.ownership is not None and self.ownership['owner'] == 'task600'
                and self.ownership['phase'] == 'recovery' and self.ledger is None
                and pending is not None and pending.target.kind == 'start_resume_current_save'
                and pending.source.scene == 'start_resume_confirm' and observation.scene == 'start_resume_home'
                and (self.handoff or {}).get('cultivation_start', {}).get('status') == 'complete'):
            return True
        if (self.ownership is not None and self.ownership['owner'] == 'task600'
                and self.ownership['phase'] == 'recovery' and self.ledger is None
                and pending is not None and pending.target.kind == 'start_ap_save_home'
                and pending.source.scene == 'start_ap_menu' and observation.scene == 'start_created'
                and (self.handoff or {}).get('cultivation_start', {}).get('status') == 'created_ap_unverified'):
            return True
        if (self._navigation_only and pending is not None and observation.scene == 'saved_home'
                and (pending.target.kind == 'date_resume_saved' or pending.target.kind == 'date_network_retry'
                    and (self.handoff or {}).get('nonbattle_context', {}).get('date_recovery', {})
                        .get('resume_requested') is True)):
            return True
        if (self.ownership is not None and self.ownership['phase'] == 'nonbattle'
                and pending is not None and pending.target.kind == 'confirm_interval_exit'
                and pending.source.scene == 'interval_end_confirm' and observation.scene == 'interval'):
            return True
        return (self.ownership is not None and self.ownership['phase'] == 'recovery'
            and pending is not None and pending.target.kind == 'recovery_notebook'
            and pending.source.scene == 'menu' and observation.scene == 'interval')

    def _decide(self, observation, config):
        from .customization_transaction import request as customization_request
        customization = customization_request(self)
        if customization is not None and customization['stage'] == 'census':
            from .customization_transaction import decide
            return decide(self, observation, config)
        from .activity_reward import request as activity_request
        if activity_request(self) is not None:
            from .activity_production import decide_activity_reward
            return decide_activity_reward(self, observation, config)
        from .rewards import active
        if active(self):
            from .reward_production import decide_reward
            return decide_reward(self, observation, config)
        if self._navigation_only:
            return self._recovery_decide(observation, config)
        if self.ownership is not None and self.ownership['phase'] == 'recovery':
            if self._recovery_decide is None:
                return Decision(Outcome.STOP, 'nonbattle_recovery_producer_not_bound')
            return self._recovery_decide(observation, config)
        page = self._page
        if page is not None and sum(menu is not None for menu in (
                page.deletion, page.drink_replacement, page.customization)) > 1:
            return Decision(Outcome.STOP, 'contradictory_nonbattle_resource_menus')
        if page is not None and page.drink_replacement is not None:
            from .drink_replacement import retained_request, plan_drink_replacement
            try:
                retained = retained_request(self)
            except ValueError:
                return Decision(Outcome.STOP, 'drink_replacement_nonbattle_scope_required')
            if (self.ledger.pending is not None or page.observation != observation
                    or page.baseline != self.ledger.state or not source_frames_match(observation, page.frame_ids)):
                return Decision(Outcome.STOP, 'drink_replacement_complete_current_baseline_required')
            plan = plan_drink_replacement(observation, self.ledger, page.drink_replacement, retained)
            if plan.effect is not None:
                effects = tuple(effect for effect in page.effects if effect.target_key != plan.effect.target_key)
                self._page = replace(page, effects=(*effects, plan.effect))
            return plan.decision
        if page is not None and page.deletion is not None:
            from .deletion import plan_deletion, retained_request
            try:
                retained = retained_request(self)
            except ValueError:
                return Decision(Outcome.STOP, 'deletion_nonbattle_scope_required')
            if (self.ledger.pending is not None or page.observation != observation
                    or page.baseline != self.ledger.state or not source_frames_match(observation, page.frame_ids)):
                return Decision(Outcome.STOP, 'deletion_complete_current_baseline_required')
            plan = plan_deletion(observation, self.ledger, page.deletion, retained)
            if plan.effect is not None:
                effects = tuple(effect for effect in page.effects if effect.target_key != plan.effect.target_key)
                self._page = replace(page, effects=(*effects, plan.effect))
            return plan.decision
        if self._nonbattle_decide is not None and (page is None or page.customization is None):
            return self._nonbattle_decide(observation, config)
        base = decide_hif(observation, config)
        page = self._page
        if (base.reason != 'card_customize_requires_verified_selection' or page is None
                or page.observation != observation or page.customization is None
                or self._customization_catalog is None):
            return base
        if (self.ledger is None or page.baseline != self.ledger.state
                or not source_frames_match(observation, page.frame_ids)):
            return Decision(Outcome.STOP, 'customization_complete_baseline_required')
        if self._customization_menu_source is not None:
            try:
                menu = self._customization_menu_source(page, config)
            except ValueError as error:
                return Decision(Outcome.STOP, str(error))
            page = self._page = replace(page, customization=menu)
        if (isinstance(page.customization, CustomizationMenu)
                and 'custom_remaining:' + page.customization.card_instance_id not in dict(self.ledger.state.counters)
                and config.mode == 'advance'):
            try:
                self.bind_observed_customization_capacity(page)
            except ValueError as error:
                return Decision(Outcome.STOP, str(error))
            page = self._page
        plan = plan_customization(observation, self.ledger, page.customization, self._customization_catalog)
        if plan.effect is not None:
            # The effect is computed from this complete menu and exact ledger;
            # a generic event/target supplied by OCR cannot select a purchase.
            effects = tuple(effect for effect in page.effects if effect.target_key != plan.effect.target_key)
            self._page = replace(page, effects=(*effects, plan.effect))
        return plan.decision

    def run(self, context, config, **kwargs):
        """Serialize Maa invocations and reload durable state before any input.

        Windows releases the byte lock when the process exits. The lock file
        remains in place so deleting/recreating it cannot bypass another host.
        """
        with locked_hif_record(self.path):
            try:
                gate = self._load()
                if self.ownership is not None and (
                        self.ownership['owner'] != self.input_owner
                        or self.ownership['game_pid'] != self.game_pid
                        or (self.ownership['phase'] != 'nonbattle'
                            and not (self.ownership['phase'] == 'recovery'
                                and self._recovery_decide is not None)
                            and not (self._navigation_only and self.ownership['phase'] == 'battle'))):
                    return Decision(Outcome.STOP, 'nonbattle_input_owner_or_phase_not_ready')
                self._page = self._confirmed_ledger = self._confirmed_reward = self._confirmed_activity = None
                self._confirmed_customization = None
                self._confirmed_recovery_drinks = None
                self._active = True
                if self._navigation_only:
                    current = self.hooks.transition_gate
                    current.pending, current.issued_at_ms, current.stopped = gate.pending, gate.issued_at_ms, gate.stopped
                    current.timeout_ms = gate.timeout_ms
                    self._prime_date_history()
                if self.ownership is not None and self._recovery_prepare:
                    gate = self._recovery_prepare(gate)
                current = self.hooks.transition_gate
                current.pending, current.issued_at_ms, current.stopped = gate.pending, gate.issued_at_ms, gate.stopped
                current.timeout_ms = gate.timeout_ms
                return run_hif_sequence(context, config, self.hooks, **kwargs)
            finally:
                self._active = False

    def reconcile_interval_confirmation(self, gate, report, proof):
        """Acknowledge the missed confirmation page, retaining the failed attempt."""
        from .supply import NAVIGATION_DESTINATIONS, navigation_effect
        pending = gate.pending
        if (not self._active or self.ownership is None or self.ownership['owner'] != 'task600'
                or self.ownership['phase'] != 'nonbattle' or pending is None
                or pending.target.kind not in NAVIGATION_DESTINATIONS
                or pending.expected_scenes != NAVIGATION_DESTINATIONS[pending.target.kind]
                or self.ledger is None or self.ledger.pending is None
                or self.ledger.pending.effect != navigation_effect(self.ledger.state, pending.target.key)
                or report.get('route', {}).get('run_id') != self.run_id
                or report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid):
            raise ValueError('interval recovery cannot discard another pending resource action')
        events = [row for row in report['route'].get('input_events', [])
            if row.get('target_key') == pending.target.key and row.get('target_kind') == pending.target.kind
            and row.get('source_revision') == pending.source.revision and row.get('issued_at_ms') == gate.issued_at_ms
            and row.get('dispatch_attempted') is True and row.get('success') is True and row.get('uncertain') is False]
        destinations = NAVIGATION_DESTINATIONS[pending.target.kind]
        if (len(events) != 1 or proof.observation.run_id != self.run_id
                or proof.observation.scene not in destinations or not source_frames_match(proof.observation, proof.frame_ids)
                or proof.observation.revision <= pending.source.revision
                or not 0 <= proof.observation.capture_age_ms + self.clock.now() - proof.observation.revision <= 1000
                or proof.receipt_frame_ids != proof.frame_ids
                or set(proof.frame_ids) & {row.get('frame_id') for row in report.get('frames', [])}):
            raise ValueError('interval confirmation lacks independent current destination evidence')
        original = transition_checkpoint(gate, self.clock.epoch)
        old_ledger, old_handoff = self.ledger, self.handoff
        settled = self.ledger.confirm(proof.receipt,
            transaction_id=self.ledger.pending.effect.transaction_id, frame_ids=proof.receipt_frame_ids,
            observation=proof.observation)
        receipts = list((self.handoff or {}).get('recovery_navigation_receipts', []))
        outcome = {'leave_interval': 'missed_interval_confirmation_observed',
            'confirm_interval_exit': 'late_battle_entry_observed', 'battle_intro_next': 'late_battle_entry_observed',
            'customization_open_list': 'late_customization_list_observed'}[pending.target.kind]
        if proof.observation.scene == 'battle_intro':
            outcome = 'late_battle_introduction_observed'
        elif proof.observation.scene == 'battle_entry':
            outcome = 'late_battle_entry_observed'
        receipts.append(dict(outcome=outcome, original_transition=original,
            original_ledger_pending=self.ledger.record()['pending'], source_result=report['id'],
            receipt_frame_ids=list(proof.frame_ids)))
        self.handoff = dict(self.handoff or {}, recovery_navigation_receipts=receipts)
        self.handoff = record_ledger_deck(self.handoff, old_ledger, settled, frames=proof.receipt_frame_ids)
        self.ledger = settled
        result = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            self._save(transition_checkpoint(result, self.clock.epoch))
        except BaseException:
            self.ledger, self.handoff = old_ledger, old_handoff
            raise
        return result

    def reconcile_inventory_entry(self, gate, report, proof, inventory):
        """Keep the successful open and accept only its fresh observed census view."""
        from .recovery_inventory import scope
        pending = gate.pending
        if (not self._active or self.ownership is None or self.ownership['phase'] != 'recovery'
                or pending is None or pending.target.kind != 'recovery_inventory'
                or pending.source.scene not in ('interval', 'start_created')
                or 'inventory' not in pending.expected_scenes
                or self.ledger is not None and self.ledger.pending is not None
                or report.get('route', {}).get('run_id') != self.run_id
                or report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid):
            raise ValueError('late inventory entry is not the original nonconsuming open')
        matching = [row for row in report['route'].get('input_events', ())
            if row.get('target_key') == pending.target.key and row.get('target_kind') == pending.target.kind
            and row.get('source_scene') == pending.source.scene and row.get('source_revision') == pending.source.revision
            and row.get('issued_at_ms') == gate.issued_at_ms and row.get('dispatch_attempted') is True
            and row.get('success') is True and row.get('uncertain') is False]
        if (len(matching) != 1 or not isinstance(proof, HifRoutePage)
                or proof.observation.run_id != self.run_id or proof.observation.scene != 'inventory'
                or validate_observation(proof.observation) is not None
                or proof.observation.revision <= pending.source.revision
                or not 0 <= proof.observation.capture_age_ms + self.clock.now() - proof.observation.revision <= 1000
                or not source_frames_match(proof.observation, proof.frame_ids)
                or set(proof.frame_ids) & {row.get('frame_id') for row in report.get('frames', ())}
                or inventory.get('displacement') != 0 or type(inventory.get('total')) is not int
                or inventory.get('recovery_scope') != scope(self)
                or inventory['total'] < 1 or not inventory.get('slots')
                or not set(proof.frame_ids).issubset(inventory.get('frame_ids', ()))
                or any(row.get('frame_ids') != list(proof.frame_ids) for row in inventory['slots'])):
            raise ValueError('late inventory entry lacks an independent current top census')
        original, previous = transition_checkpoint(gate, self.clock.epoch), self.handoff
        receipts = list((previous or {}).get('recovery_navigation_receipts', ()))
        receipts.append(dict(outcome='late_readonly_inventory_entry_observed', original_transition=original,
            source_result=report['id'], receipt_frame_ids=list(proof.frame_ids)))
        self.handoff = dict(previous or {}, recovery_navigation_receipts=receipts, nonbattle_inventory=inventory)
        result = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            self._save(transition_checkpoint(result, self.clock.epoch))
        except BaseException:
            self.handoff = previous
            raise
        return result

    def reconcile_inventory_scroll(self, gate, report, proof, inventory):
        """Settle a read-only scroll only from its original overlapping pixels."""
        pending = gate.pending
        if (not self._active or self.ownership is None or self.ownership['phase'] != 'recovery'
                or pending is None or pending.target.kind != 'recovery_inventory_scroll'
                or self.ledger is not None and self.ledger.pending is not None
                or report.get('route', {}).get('run_id') != self.run_id
                or report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid):
            raise ValueError('late inventory recovery is not the retained nonconsuming scroll')
        matching = [row for row in report['route'].get('input_events', [])
            if row.get('target_key') == pending.target.key and row.get('target_kind') == pending.target.kind
            and row.get('source_revision') == pending.source.revision and row.get('issued_at_ms') == gate.issued_at_ms
            and row.get('dispatch_attempted') is True and row.get('success') is True and row.get('uncertain') is False]
        sources = [event for event in report['route'].get('events', []) if event.get('phase') == 'inventory'
            and source_frames_match(pending.source, event.get('frame_ids', ()))
            and int(event['frame_ids'][-1].rsplit('-', 1)[1]) + event['capture_age_ms'] == pending.source.revision]
        old = (self.handoff or {}).get('nonbattle_inventory', {})
        registration = inventory.get('registration') or {}
        shifts = registration.get('shifts', [])
        anchors = registration.get('anchor_indices', [])
        old_rows = {row['index']: row for row in old.get('slots', [])}
        current = {row['index']: row for row in inventory.get('slots', [])}
        if (len(matching) != 1 or len(sources) != 1 or len(shifts) != 1
                or any(type(delta) is not int or not 1 <= delta <= 450 for delta in shifts)
                or registration.get('source_displacement') != old.get('displacement')
                or inventory.get('displacement') != old['displacement'] + shifts[0]
                or registration.get('source_frame_ids') != sources[0]['frame_ids']
                or registration.get('receipt_frame_ids') != list(proof.frame_ids)
                or len(anchors) < 2 or any(index not in old_rows for index in anchors)
                or len({old_rows[index]['box'][0] for index in anchors}) < 2
                or inventory.get('total') != old.get('total')
                or any(index in old_rows and row['card_id'] != old_rows[index]['card_id']
                    for index, row in current.items())
                or proof.observation.run_id != self.run_id or proof.observation.scene != 'inventory'
                or validate_observation(proof.observation) is not None or proof.observation.page_id == pending.source.page_id
                or proof.observation.revision <= pending.source.revision
                or not 0 <= proof.observation.capture_age_ms + self.clock.now() - proof.observation.revision <= 1000
                or not source_frames_match(proof.observation, proof.frame_ids)
                or set(proof.frame_ids) & {row.get('frame_id') for row in report.get('frames', [])}):
            raise ValueError('late inventory scroll lacks independent measured overlap')
        original = transition_checkpoint(gate, self.clock.epoch)
        previous = self.handoff
        receipts = list((previous or {}).get('recovery_navigation_receipts', []))
        receipts.append(dict(outcome='late_readonly_inventory_scroll_observed', original_transition=original,
            source_result=report['id'], receipt_frame_ids=list(proof.frame_ids), registration=registration))
        self.handoff = dict(previous or {}, recovery_navigation_receipts=receipts, nonbattle_inventory=inventory)
        result = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            self._save(transition_checkpoint(result, self.clock.epoch))
        except BaseException:
            self.handoff = previous
            raise
        return result

    def bind_recovered_ledger(self, ledger, *, frame_ids):
        """Complete recovery under the already-held route lock, before yielding."""
        gate = self.hooks.transition_gate
        page = self._page
        if (not self._active or self.ownership is None or self.ownership['owner'] != 'task600'
                or self.ownership['phase'] != 'recovery' or gate.pending is not None or gate.stopped
                or not isinstance(ledger, HifLedger) or ledger.pending is not None
                or ledger.state.run_id != self.run_id or page is None or page.baseline != ledger.state
                or page.frame_ids != frame_ids or not source_frames_match(page.observation, frame_ids)
                or not 0 <= page.observation.capture_age_ms + self.clock.now() - page.observation.revision <= 1000):
            raise ValueError('current complete nonbattle recovery baseline is not independently bound')
        old_ledger, old_owner = self.ledger, self.ownership
        handoff = self.handoff or {}
        old_frames = (*handoff.get('source_frame_ids', ()),
            *handoff.get('terminal_receipt', {}).get('frame_ids', ()))
        if (set(frame_ids) & set(old_frames) or old_ledger is not None and (
                old_ledger.pending is not None or ledger.state.revision <= old_ledger.state.revision)):
            raise ValueError('current recovery cannot reuse old evidence or discard pending')
        old_handoff = self.handoff
        self.ledger = ledger
        self.ownership = dict(old_owner, phase='nonbattle', revision=old_owner['revision'] + 1)
        self.handoff = dict(handoff, recovery_frame_ids=list(frame_ids))
        try:
            self.handoff = record_ledger_deck(self.handoff, old_ledger, ledger, frames=frame_ids,
                opening=old_ledger is None and page.observation.scene == 'start_created', census=True)
            self._save(transition_checkpoint(gate, self.clock.epoch))
        except BaseException:
            self.ledger, self.ownership, self.handoff = old_ledger, old_owner, old_handoff
            raise

    def reconcile_inventory_detail(self, gate, report, proof, inventory):
        """Accept a late nonconsuming detail receipt, retaining the old attempt."""
        pending = gate.pending
        if (not self._active or self.ownership is None or self.ownership['phase'] != 'recovery'
                or pending is None or pending.target.kind != 'recovery_inventory_detail'
                or self.ledger is not None and self.ledger.pending is not None
                or not isinstance(report, dict) or report.get('route', {}).get('run_id') != self.run_id
                or report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid):
            raise ValueError('late inventory recovery is not the retained read-only detail')
        events = report['route'].get('input_events', [])
        matching = [row for row in events if row.get('target_key') == pending.target.key
            and row.get('source_revision') == pending.source.revision
            and row.get('issued_at_ms') == gate.issued_at_ms and row.get('success') is True
            and row.get('dispatch_attempted') is True and row.get('uncertain') is False]
        index = int(pending.target.key.split(':')[1])
        old = {row['index']: row for row in (self.handoff or {}).get('nonbattle_inventory', {}).get('slots', [])}
        current = {row['index']: row for row in inventory.get('slots', [])}
        row = current.get(index, {})
        old_frames = {row.get('frame_id') for row in report.get('frames', [])}
        if (len(matching) != 1 or index not in old or not old[index].get('card_id')
                or row.get('card_id') != old[index]['card_id'] or 'customizations' not in row
                or row.get('detail', {}).get('frame_ids') != list(proof.frame_ids)
                or proof.observation.scene != 'inventory' or proof.observation.run_id != self.run_id
                or proof.observation.page_id == pending.source.page_id
                or validate_observation(proof.observation) is not None or proof.observation.revision <= pending.source.revision
                or not 0 <= proof.observation.capture_age_ms + self.clock.now() - proof.observation.revision <= 1000
                or not source_frames_match(proof.observation, proof.frame_ids) or set(proof.frame_ids) & old_frames
                or inventory.get('detail_rejection') is not None):
            raise ValueError('late inventory detail lacks independent exact current evidence')
        retained = (self.handoff or {}).get('nonbattle_inventory', {})
        if (inventory.get('total') != retained.get('total') or inventory.get('displacement') != 0
                or retained.get('displacement') != 0 or set(current) != set(old)
                or any(key != index and (current[key].get('current_card_id') != value['card_id']
                    or current[key].get('presence') != value['presence']
                    or current[key].get('frame_ids') != list(proof.frame_ids)) for key, value in old.items())):
            raise ValueError('late inventory detail did not recheck the same unscrolled physical list')
        original = transition_checkpoint(gate, self.clock.epoch)
        previous = self.handoff
        receipts = list((previous or {}).get('recovery_navigation_receipts', []))
        receipts.append(dict(outcome='late_readonly_inventory_detail_observed', original_transition=original,
            source_result=report['id'], receipt_frame_ids=list(proof.frame_ids), card_index=index,
            card_id=row['card_id'], receipt_revision=proof.observation.revision))
        self.handoff = dict(previous or {}, recovery_navigation_receipts=receipts, nonbattle_inventory=inventory)
        result = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            self._save(transition_checkpoint(result, self.clock.epoch))
        except BaseException:
            self.handoff = previous
            raise
        return result

    def reconcile_ending_navigation(self, gate, report, proof, *, terminal_report=None):
        """Keep a failed closing input verbatim and acknowledge its late destination."""
        from .ending import (
            DEFEAT_EXIT_DESTINATIONS,
            STAR_REWARD_DESTINATIONS,
            MEMORY_DETAIL_DESTINATIONS,
            FINAL_EVALUATION_DESTINATIONS,
            ROUND_TWO_RESULT_DESTINATIONS,
            MEMORY_COMPLETION_DESTINATIONS,
            MEMORY_GENERATION_DESTINATIONS,
            PRODUCE_COMPLETION_DESTINATIONS,
            same_closing_page,
            verified_ending_receipt,
            star_reward_destinations,
        )
        receipt = verified_ending_receipt(self, terminal_report=terminal_report,
            round_number=(self.handoff or {}).get('round_number'))
        if terminal_report is not None:
            self._ending_terminal_report = terminal_report
        pending = gate.pending
        contracts = {'end_result_next': ('end_result', ROUND_TWO_RESULT_DESTINATIONS),
            'end_defeat_exit': ('ending_exit_confirmation', DEFEAT_EXIT_DESTINATIONS),
            'end_star_reward_next': ('ending_summary', star_reward_destinations(receipt['round_number'])),
            'end_final_evaluation_next': ('ending_summary', FINAL_EVALUATION_DESTINATIONS),
            'end_generate_memory': ('ending_generation', MEMORY_GENERATION_DESTINATIONS),
            'end_memory_complete_next': ('ending_generated_memory', MEMORY_COMPLETION_DESTINATIONS),
            'end_memory_detail_next': ('ending_complete', MEMORY_DETAIL_DESTINATIONS),
            'end_produce_complete_next': ('ending_summary', PRODUCE_COMPLETION_DESTINATIONS)}
        if receipt['round_number'] == 1:
            contracts['round_one_award_next'] = ('round_one_award', ('interval', 'round_one_award', *STAR_REWARD_DESTINATIONS))
        contract = contracts.get(pending.target.kind) if pending is not None else None
        if (not self._active or pending is None or contract is None
                or pending.source.scene != contract[0] or pending.expected_scenes != contract[1]
                or pending.target.kind == 'round_one_award_next' and pending.source.page_id.count(':') != 6
                or pending.target.kind == 'end_final_evaluation_next'
                    and pending.source.page_id != 'recovery:ending_summary:::::final_evaluation'
                or pending.target.kind == 'end_result_next' and receipt['round_number'] != 2
                or pending.target.kind == 'end_defeat_exit' and receipt.get('outcome') != 'defeat'
                or self.ledger is not None and self.ledger.pending is not None
                or not isinstance(report, dict) or not isinstance(report.get('id'), str)
                or report.get('route', {}).get('run_id') != self.run_id
                or report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid):
            raise ValueError('ending recovery is not this accepted result navigation')
        matching = [row for row in report['route'].get('input_events', [])
            if row.get('target_key') == pending.target.key and row.get('target_kind') == pending.target.kind
            and row.get('source_scene') == pending.source.scene
            and row.get('source_revision') == pending.source.revision
            and row.get('issued_at_ms') == gate.issued_at_ms
            and row.get('dispatch_attempted') is True and row.get('success') is True
            and row.get('uncertain') is False]
        round_one_item = (receipt['round_number'] == 1 and pending.target.kind == 'end_star_reward_next'
            and isinstance(proof, HifRoutePage) and proof.observation.scene == 'round_one_award'
            and (proof.closing or {}).get('phase') == 'round_one_award'
            and isinstance((proof.closing or {}).get('award_name'), str)
            and bool(proof.closing['award_name']))
        if (len(matching) != 1 or not isinstance(proof, HifRoutePage)
                or validate_observation(proof.observation) is not None
                or proof.observation.state_key == pending.source.state_key and not current_completion_next(pending, proof.observation)
                or same_closing_page(pending.source, proof.observation, proof.closing or {})
                or proof.observation.scene not in pending.expected_scenes
                or proof.observation.scene == 'ending_exit_confirmation' and receipt.get('outcome') != 'defeat'
                or proof.observation.run_id != self.run_id
                or (proof.observation.scenario, proof.observation.idol, proof.observation.plan)
                    != (pending.source.scenario, pending.source.idol, pending.source.plan)
                or not source_frames_match(proof.observation, proof.frame_ids) or proof.observation.revision <= pending.source.revision
                or not 0 <= proof.observation.capture_age_ms + self.clock.now() - proof.observation.revision <= self._input_age_limit(pending)
                or any(not isinstance(key, str) or not key for key in proof.frame_ids)
                or set(proof.frame_ids) & (set(receipt['frame_ids'])
                    | {row.get('frame_id') for row in report.get('frames', [])})):
            raise ValueError('ending recovery lacks independent current expected-page evidence')
        original = transition_checkpoint(gate, self.clock.epoch)
        previous = self.handoff
        previous_page = self._page
        receipts = list((previous or {}).get('recovery_navigation_receipts', []))
        receipts.append(dict(outcome=('late_round_one_award_observed' if round_one_item
                else 'late_round_two_result_destination_observed' if pending.target.kind == 'end_result_next'
                else 'late_closing_navigation_destination_observed'),
            original_transition=original, source_result=report['id'],
            receipt_frame_ids=list(proof.frame_ids), receipt_revision=proof.observation.revision,
            destination=proof.observation.scene))
        if round_one_item:
            receipts[-1]['obtained_award'] = dict(name=proof.closing['award_name'],
                description=proof.closing.get('award_description'))
        self.handoff = dict(previous or {}, recovery_navigation_receipts=receipts)
        if pending.target.kind == 'end_generate_memory' and proof.observation.scene == 'ending_generated_memory':
            from .closing_state import complete_memory
            try:
                self._page = proof
                self.handoff['closing'] = complete_memory(self, pending, gate.issued_at_ms, proof,
                    report=report, terminal_report=terminal_report)
            except BaseException:
                self.handoff = previous
                self._page = previous_page
                raise
        settled = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            self._save(transition_checkpoint(settled, self.clock.epoch))
        except BaseException:
            self.handoff = previous
            self._page = previous_page
            raise
        return settled

    def accept_ending_home(self, proof, *, terminal_report=None):
        from .closing_state import complete_home
        closing = complete_home(self, proof, terminal_report=terminal_report)
        if (self.handoff or {}).get('closing') == closing:
            return
        previous = self.handoff
        self.handoff = dict(previous or {}, closing=closing)
        try:
            self._save(transition_checkpoint(self.hooks.transition_gate, self.clock.epoch))
        except BaseException:
            self.handoff = previous
            raise

    def accept_start_ap(self, proof, reports, *, observed_report=None):
        from .ap_readback import verified_ap_receipt
        state = verified_ap_receipt(self, proof, reports, observed_report=observed_report)
        previous = self.handoff
        self.handoff = dict(previous or {}, cultivation_start=state)
        try:
            self._save(transition_checkpoint(self.hooks.transition_gate, self.clock.epoch))
        except BaseException:
            self.handoff = previous
            raise

    def reconcile_round_one_return(self, gate, source_report, proof, resources):
        from .ending import ROUND_ONE_RESULT_DESTINATIONS, verified_ending_receipt
        pending = gate.pending
        if (not self._active or self.ownership is None or self.ownership['phase'] != 'recovery'
                or pending is None or pending.target.kind != 'round_one_result_next'
                or pending.source.scene != 'round_one_result'
                or pending.expected_scenes != ROUND_ONE_RESULT_DESTINATIONS
                or self.ledger is not None and self.ledger.pending is not None
                or source_report.get('route', {}).get('run_id') != self.run_id
                or source_report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid):
            raise ValueError('round-one return is not the original submitted nonconsuming result acknowledgement')
        receipt = verified_ending_receipt(self, terminal_report=self._ending_terminal_report, round_number=1)
        matching = [event for event in source_report['route'].get('input_events', ())
            if event.get('target_kind') == pending.target.kind and event.get('target_key') == pending.target.key
            and event.get('source_revision') == pending.source.revision and event.get('issued_at_ms') == gate.issued_at_ms
            and event.get('dispatch_attempted') is True and event.get('success') is True and event.get('uncertain') is False]
        interval = (isinstance(resources, tuple) and len(resources) == 3
            and all(type(value) is int for value in resources)
            and 0 <= resources[0] <= resources[1] <= 100 and resources[1] >= 1 and resources[2] >= 0
            and proof.observation.scene == 'interval'
            and proof.observation.page_id.startswith('recovery:interval:' + ':'.join(map(str, resources)) + ':'))
        reward = proof.closing or {}
        star_reward = (proof.observation.scene == 'ending_summary' and reward.get('phase') == 'ending_summary'
            and reward.get('summary_kind') == 'star_reward'
            and type(reward.get('star_total')) is int and type(reward.get('star_increment')) is int
            and 0 <= reward['star_increment'] <= reward['star_total']
            and proof.observation.page_id == f'recovery:ending_summary:::::star_reward:{reward["star_total"]}:{reward["star_increment"]}'
            and len(proof.observation.targets) == 1 and proof.observation.targets[0].kind == 'end_star_reward_next')
        if (len(matching) != 1 or not (interval or star_reward)
                or proof.observation.run_id != self.run_id
                or validate_observation(proof.observation) is not None
                or proof.observation.revision <= pending.source.revision
                or not 0 <= proof.observation.capture_age_ms + self.clock.now() - proof.observation.revision <= 1000
                or not source_frames_match(proof.observation, proof.frame_ids)
                or set(proof.frame_ids).intersection(frame.get('frame_id') for frame in source_report.get('frames', ()))
                or set(proof.frame_ids).intersection(receipt['frame_ids'])):
            raise ValueError('round-one return lacks an independent complete current interval or star reward')
        original, previous = transition_checkpoint(gate, self.clock.epoch), self.handoff
        result = TransitionGate(timeout_ms=gate.timeout_ms)
        receipts = list((self.handoff or {}).get('recovery_navigation_receipts', ()))
        receipts.append(dict(outcome=('late_verified_round_one_star_reward' if star_reward
                else 'late_verified_round_one_result_return'), source_result=source_report['id'],
            original_transition=original, receipt_frame_ids=list(proof.frame_ids)))
        self.handoff = dict(self.handoff or {}, recovery_navigation_receipts=receipts)
        try:
            self._save(transition_checkpoint(result, self.clock.epoch))
        except BaseException:
            self.handoff = previous
            raise
        return result

    def reconcile_recovery_drink(self, gate, report, proof):
        from .recovery_drinks import KINDS, confirm
        pending = gate.pending
        if (not self._active or self.ownership is None or self.ownership['phase'] != 'recovery'
                or pending is None or pending.target.kind not in KINDS
                or self.ledger is not None and self.ledger.pending is not None
                or report.get('route', {}).get('run_id') != self.run_id
                or report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid
                or transition_checkpoint(gate, self.clock.epoch) != transition_checkpoint(self.hooks.transition_gate, self.clock.epoch)):
            raise ValueError('late recovery drink receipt is not its original owned read-only input')
        matching = [event for event in report['route'].get('input_events', ())
            if event.get('target_kind') == pending.target.kind and event.get('target_key') == pending.target.key
            and event.get('source_revision') == pending.source.revision and event.get('issued_at_ms') == gate.issued_at_ms
            and event.get('dispatch_attempted') is True and event.get('success') is True and event.get('uncertain') is False]
        seen = proof.observation
        if (len(matching) != 1 or seen.run_id != self.run_id or seen.scene not in pending.expected_scenes
                or validate_observation(seen) is not None or seen.revision <= pending.source.revision
                or not 0 <= seen.capture_age_ms + self.clock.now() - seen.revision <= 1000
                or not source_frames_match(seen, proof.frame_ids)
                or set(proof.frame_ids).intersection(frame.get('frame_id') for frame in report.get('frames', ()))):
            raise ValueError('late recovery drink receipt lacks its independent current destination')
        previous = self.handoff, self._page, self._confirmed_recovery_drinks
        original = transition_checkpoint(gate, self.clock.epoch)
        result = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            self._page = proof
            self._confirmed_recovery_drinks = confirm(self, proof)
            receipts = list((self.handoff or {}).get('recovery_navigation_receipts', ()))
            receipts.append(dict(outcome='late_verified_recovery_drink', source_result=report['id'],
                original_transition=original, receipt_frame_ids=list(proof.frame_ids)))
            self.handoff = dict(self.handoff or {}, recovery_navigation_receipts=receipts)
            self._save(transition_checkpoint(result, self.clock.epoch))
        except BaseException:
            self.handoff, self._page, self._confirmed_recovery_drinks = previous
            raise
        return result

    def reconcile_recovery_navigation(self, gate, source_report, proof, resources):
        """Reconcile one stopped read-only navigation after a proven late arrival.

        This never applies to a resource action or ledger pending. The old
        stopped transition/deadline is archived verbatim, not renewed/succeeded.
        """
        pending = gate.pending
        if (not self._active or self.ownership is None or self.ownership['phase'] != 'recovery'
                or not gate.stopped or pending is None or pending.target.kind != 'recovery_notebook'
                or (self.ledger is not None and self.ledger.pending is not None)
                or not isinstance(source_report, dict)
                or source_report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid
                or source_report.get('route', {}).get('run_id') != self.run_id
                or source_report['route'].get('decision', {}).get('reason') != 'unexpected_destination'):
            raise ValueError('stopped HIF resource/unknown input cannot be reconciled as navigation')
        frames = source_report.get('frames', [])
        if (not frames or any(row.get('phase') != 'interval' or tuple(
                row.get('observation', {}).get(k) for k in ('stamina', 'max_stamina', 'currency')) != resources
                for row in (frames[0], frames[-1]))
                or not isinstance(proof, HifRoutePage) or proof.observation.scene not in pending.expected_scenes
                or not proof.observation.page_id.startswith('recovery:' + proof.observation.scene + ':' + ':'.join(map(str, resources)) + ':')
                or proof.observation.run_id != self.run_id or validate_observation(proof.observation) is not None
                or proof.observation.revision <= pending.source.revision
                or proof.observation.capture_age_ms < 0
                or proof.observation.capture_age_ms + self.clock.now() - proof.observation.revision > 1000
                or not source_frames_match(proof.observation, proof.frame_ids)
                or set(proof.frame_ids) & {row.get('frame_id') for row in frames}):
            raise ValueError('late HIF navigation requires independent expected notebook evidence')
        matched = []
        for event in source_report['route'].get('events', []):
            keys = event.get('frame_ids', [])
            if event.get('phase') == pending.source.scene and source_frames_match(pending.source, keys):
                try:
                    revision = int(keys[-1].rsplit('-', 1)[1]) + event['capture_age_ms']
                except (ValueError, TypeError, KeyError):
                    continue
                if revision == pending.source.revision:
                    matched.append(event)
        if (len(matched) != 1 or not any(row.get('phase') == pending.source.scene
                and row.get('frame_id') == matched[0]['frame_ids'][0] for row in frames[1:-1])):
            raise ValueError('HIF failed navigation source does not match retained pending')
        original = transition_checkpoint(gate, self.clock.epoch)
        old_handoff = self.handoff
        receipts = list((self.handoff or {}).get('recovery_navigation_receipts', ()))
        receipts.append(dict(outcome='late_readonly_navigation_destination_observed',
            original_transition=original, source_result=source_report['id'],
            source_frame_ids=matched[0]['frame_ids'], receipt_frame_ids=list(proof.frame_ids),
            receipt_revision=proof.observation.revision, resources=list(resources)))
        self.handoff = dict(self.handoff or {}, recovery_navigation_receipts=receipts)
        settled = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            self._save(transition_checkpoint(settled, self.clock.epoch))
        except BaseException:
            self.handoff = old_handoff
            raise
        return settled

    def _load(self):
        raw = json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(raw, dict) or type(raw.get('schema_version')) is not int:
            raise ValueError('HIF host checkpoint identity/version mismatch')
        version = raw['schema_version']
        keys = {'schema_version', 'run_id', 'transition'} | ({'ledger'} if version in (2, 3) else set())
        if version == 3:
            keys |= {'ownership', 'handoff'}
        if version not in (1, 2, 3) or set(raw) != keys or raw['run_id'] != self.run_id:
            raise ValueError('HIF host checkpoint identity/version mismatch')
        gate = restore_transition(raw['transition'], clock_epoch=self.clock.epoch)
        if version == 3:
            self.ownership = validate_ownership(raw['ownership'])
            self.handoff = raw['handoff']
        elif self.ownership is not None:
            raise ValueError('HIF legacy record cannot acquire ownership implicitly')
        if version in (2, 3) and raw['ledger'] is not None:
            if self._catalog is None:
                raise ValueError('HIF ledger catalog binding missing')
            ledger = restore_ledger(raw['ledger'], card_catalog=self._catalog, drink_ids=self._drink_ids)
            if ledger.state.run_id != self.run_id:
                raise ValueError('HIF ledger belongs to another run')
            recovery_navigation = (version == 3 and self.ownership['phase'] == 'recovery'
                and gate.pending is not None and gate.pending.target.kind in RECOVERY_NAVIGATION
                and ledger.pending is None)
            borrowed_date = False
            if (version == 3 and not self._navigation_only and self.input_owner == 'task600' and not self._active
                    and self.ownership['owner'] == 'task275' and self.ownership['phase'] == 'battle'
                    and gate.pending is not None and gate.pending.target.kind in SAVED_CULTIVATION_NAVIGATION
                    and ledger.pending is None):
                from .input_receipt import original_input_report, cancelled_unissued_report
                # Bootstrap constructs a nonbattle reader before scheduling
                # the actual owner. This receipt allows loading only; run()
                # still rejects its owner mismatch before any observation.
                carrier = (self.handoff or {}).get('nonbattle_context', {}).get('route_input_receipt') or {}
                if carrier.get('dispatch_state') == 'succeeded':
                    original_input_report(self, gate)
                else:
                    cancelled_unissued_report(self, gate)
                borrowed_date = True
            recovery_navigation = recovery_navigation or (version == 3 and self._navigation_only
                and self.ownership['owner'] == 'task275' and self.ownership['phase'] == 'battle'
                and gate.pending is not None and gate.pending.target.kind in SAVED_CULTIVATION_NAVIGATION
                and ledger.pending is None)
            recovery_navigation = recovery_navigation or borrowed_date
            from .rewards import permits_navigation
            from .activity_reward import permits_navigation as permits_activity
            from .customization_transaction import permits_navigation as permits_customization
            reward_navigation = version == 3 and (permits_navigation(self, ledger, gate)
                or permits_activity(self, ledger, gate) or permits_customization(self, ledger, gate))
            if (((ledger.pending is None) != (gate.pending is None) and not recovery_navigation)
                    or (ledger.pending is not None and (
                        gate.pending is None or gate.pending.target.key != ledger.pending.effect.target_key
                        or gate.pending.source.day != ledger.state.day))) and not reward_navigation:
                raise ValueError('HIF ledger and route pending disagree')
            self.ledger = ledger
        elif version == 3 and (self.ownership['owner'] != 'task600'
                or self.ownership['phase'] != 'recovery'):
            raise ValueError('HIF active input owner requires a complete ledger')
        elif self.ledger is not None:
            raise ValueError('HIF legacy route checkpoint has no ledger; cannot infer a baseline')
        return gate

    def _confirmation_ready(self, observation):
        self._confirmed_ledger = None
        self._confirmed_reward = None
        self._confirmed_activity = None
        self._confirmed_customization = None
        self._confirmed_recovery_drinks = None
        from .recovery_drinks import KINDS, confirm
        route = self.hooks.transition_gate.pending
        from .prestart_search import MEMORY_SEARCH_CONTRACTS, search_destination
        if route is not None and route.target.kind in MEMORY_SEARCH_CONTRACTS:
            return (self._page is not None and self._page.observation == observation
                and search_destination(route, observation, self._page.start))
        if route is not None and route.target.kind in KINDS:
            if self._page is None or self._page.observation != observation:
                return False
            self._confirmed_recovery_drinks = confirm(self, self._page)
            return True
        if self.ledger is None or self.ledger.pending is None:
            return True
        page = self._page
        from .customization_transaction import request as customization_request
        from .customization_transaction import confirm_page as confirm_customization
        if customization_request(self) is not None and page is not None and page.observation == observation:
            self._confirmed_customization = confirm_customization(self, page)
            return True
        from .activity_reward import request as activity_request
        from .activity_reward import confirm_page as confirm_activity
        from .activity_reward import support_source
        if (support_source(route) is not None and page is not None
                and page.observation == observation and observation.scene == 'support_reward'):
            from .activity_reward import start_support_request
            self._confirmed_activity = start_support_request(self, page)
            return True
        if activity_request(self) is not None and page is not None and page.observation == observation and page.activity is not None:
            self._confirmed_activity = confirm_activity(self, page)
            return True
        if (route is not None and route.target.kind == 'finish_consult' and page is not None
                and page.observation == observation and observation.scene == 'reward_delete_choice'
                and (page.reward or {}).get('kind') == 'delete_choice'):
            from .rewards import consult_exit_request
            self._confirmed_reward = dict(consult_exit_request(self, self.hooks.transition_gate),
                deletion=deepcopy(page.reward['deletion']))
            return True
        from .rewards import request, confirm_page
        reward = request(self)
        if reward is not None and page is not None and page.observation == observation and page.reward is not None:
            self._confirmed_reward = confirm_page(self, page)
            return True
        if page is None or page.observation != observation or page.receipt is None:
            return False
        route = self.hooks.transition_gate.pending
        if route is not None and route.target.kind == 'execute_customization':
            effect = self.ledger.pending.effect
            result = page.customization_result
            if (len(effect.remove_cards) != 1 or not isinstance(result, CustomizationResult)
                    or result.card_instance_id != effect.remove_cards[0].instance_id
                    or result.card_id != effect.remove_cards[0].card_id
                    or result.frame_ids != page.receipt_frame_ids):
                return False
        self._confirmed_ledger = self.confirm_page_ledger(page)
        return True

    def confirm_page_ledger(self, page):
        ledger, pending = self.ledger, self.ledger.pending
        if (page.observation.scene == 'lesson_choice' and ledger.state.day == 1
                and ledger.state.phase == 'schedule' and pending.effect.target_key == 'lesson:Vi'):
            from .daily import lesson_selection_effect
            if pending.effect != lesson_selection_effect(ledger.state):
                raise ValueError('first lesson entry differs from its original selection')
            # The same three taps may leave SELECT visible or enter the lesson.
            # Only an observed classroom changes the phase. It costs no stamina.
            ledger = HifLedger(ledger.state, card_catalog=ledger.card_catalog,
                drink_ids=ledger.drink_ids, settled=ledger.settled).prepare(
                    replace(pending.effect, phase='lesson_choice'), frame_ids=pending.source_frame_ids)
        return ledger.confirm(page.receipt, transaction_id=pending.effect.transaction_id,
            frame_ids=page.receipt_frame_ids, observation=page.observation)

    def _cancel_unissued(self):
        gate = TransitionGate(timeout_ms=self.hooks.transition_gate.timeout_ms)
        self._save(transition_checkpoint(gate, self.clock.epoch), cancel_unissued=True)
        current = self.hooks.transition_gate
        current.pending, current.issued_at_ms, current.stopped = None, None, False
        return True

    def bind_observed_customization_capacity(self, page):
        """Publish a menu observation under the existing run lock, without input."""
        from .customization import observe_customization_capacity
        gate, owner = self.hooks.transition_gate, self.ownership or {}
        if (not self._active or page is not self._page or self.ledger is None
                or owner.get('owner') != 'task600' or owner.get('phase') != 'nonbattle'
                or owner.get('game_pid') != self.game_pid or self._navigation_only
                or gate.pending is not None or gate.stopped or self.ledger.pending is not None
                or page.baseline != self.ledger.state or page.observation.scene != 'card_customize'
                or (page.observation.idol, page.observation.plan) != active_hif_role(self)
                or page.observation.scenario != 'HIF'
                or not isinstance(page.customization, CustomizationMenu)
                or validate_observation(page.observation) is not None
                or page.observation.capture_age_ms > 1000
                or not 0 <= self.clock.now() - page.observation.revision <= 1000
                or not source_frames_match(page.observation, page.frame_ids)):
            raise ValueError('customization initial capacity needs a fresh complete current menu')
        captures = [re.fullmatch(r'route-frame-' + str(self.game_pid) + r'-([0-9]+)', key)
            if isinstance(key, str) else None for key in page.frame_ids]
        if (not all(captures) or [int(match[1]) for match in captures] != sorted(int(match[1]) for match in captures)
                or int(captures[-1][1]) > page.observation.revision
                or not 0 <= self.clock.now() - int(captures[0][1]) <= 1000):
            raise ValueError('customization capacity frames do not belong to the bound current game')
        candidate = observe_customization_capacity(page.observation, self.ledger,
            page.customization, self._customization_catalog)
        if candidate is self.ledger:
            return False
        handoff = deepcopy(self.handoff or {})
        history = handoff.setdefault('nonbattle_context', {})
        proofs = history.setdefault('customization_capacity', {})
        instance = page.customization.card_instance_id
        if not isinstance(proofs, dict) or instance in proofs:
            raise ValueError('customization capacity cannot recreate a missing historical counter')
        original = self._prepared_business_record()
        if json.loads(self.path.read_text(encoding='utf-8')) != original:
            raise ValueError('customization capacity source record changed')
        proofs[instance] = dict(schema_version=1, card_instance_id=instance, card_id=page.customization.card_id,
            source_revision=self.ledger.state.revision, ledger_revision=candidate.state.revision,
            total_remaining=page.customization.total_remaining,
            remaining=[list(row) for row in page.customization.remaining], frame_ids=list(page.frame_ids))
        envelope = dict(original, ledger=candidate.record(), handoff=handoff)
        write_hif_record(self.path, envelope)
        self.ledger, self.handoff = candidate, handoff
        self._page = replace(page, baseline=candidate.state)
        return True

    def reconcile_customization_navigation(self, gate, report, proof):
        """Confirm a late destination; never replay its original input."""
        from .input_receipt import original_input_report
        from .customization_transaction import request, confirm_page
        if (not self._active or request(self) is None or gate.pending is None
                or transition_checkpoint(gate, self.clock.epoch)
                    != transition_checkpoint(self.hooks.transition_gate, self.clock.epoch)
                or original_input_report(self, gate) != report
                or proof.observation.revision <= gate.pending.source.revision):
            raise ValueError('late customization destination has a different original submission')
        previous = self.handoff, self._page, self._confirmed_customization
        receipts = list((self.handoff or {}).get('recovery_navigation_receipts', ()))
        receipts.append(dict(outcome='late_verified_customization_destination', source_result=report['id'],
            original_transition=transition_checkpoint(gate, self.clock.epoch),
            original_ledger_pending=self.ledger.record()['pending'], receipt_frame_ids=list(proof.frame_ids)))
        candidate = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            self._page = proof
            self._confirmed_customization = confirm_page(self, proof)
            self.handoff = dict(self.handoff or {}, recovery_navigation_receipts=receipts)
            self._save(transition_checkpoint(candidate, self.clock.epoch))
        except BaseException:
            self.handoff, self._page, self._confirmed_customization = previous
            raise
        return candidate

    def bind_observed_customization_receipt(self, page):
        """Settle the original submission only after census and current menu."""
        from .customization_transaction import resolve, with_request
        if (not self._active or page is not self._page or self.ledger is None or self.ledger.pending is None
                or self.hooks.transition_gate.pending is not None or self.hooks.transition_gate.stopped):
            raise ValueError('customization settlement requires its active completed navigation')
        original = self._prepared_business_record()
        if json.loads(self.path.read_bytes()) != original:
            raise ValueError('customization settlement source record changed')
        ledger, receipt = resolve(self, page)
        handoff = with_request(self.handoff, receipt)
        handoff = record_ledger_deck(handoff, self.ledger, ledger, frames=page.frame_ids)
        write_hif_record(self.path, dict(original, ledger=ledger.record(), handoff=handoff))
        self.ledger, self.handoff = ledger, handoff
        self._page = replace(page, baseline=ledger.state)
        return ledger

    def bind_consult_exit_opening_census(self, report):
        """Consume saved static census evidence; never grant or dispatch input."""
        from .rewards import with_request, resolve_consult_exit_opening

        if self._active:
            raise ValueError('saved opening census cannot race an active route')
        with locked_hif_record(self.path):
            gate = self._load()
            self.hooks = replace(self.hooks, transition_gate=gate)
            ledger, reward = resolve_consult_exit_opening(self, report)
            handoff = with_request(self.handoff, reward)
            handoff = record_ledger_deck(handoff, self.ledger, ledger, frames=reward['resolution']['frame_ids'])
            result = TransitionGate(timeout_ms=gate.timeout_ms)
            receipts = list(handoff.get('recovery_navigation_receipts', ()))
            receipts.append(dict(outcome='verified_consult_deletion_at_battle_opening',
                source_result=report['id'], original_transition=transition_checkpoint(gate, self.clock.epoch),
                original_ledger_pending=self.ledger.record()['pending']))
            handoff['recovery_navigation_receipts'] = receipts
            write_hif_record(self.path, dict(schema_version=3, run_id=self.run_id,
                transition=transition_checkpoint(result, self.clock.epoch), ledger=ledger.record(),
                ownership=self.ownership, handoff=handoff))
            self.ledger, self.handoff = ledger, handoff
            self.hooks = replace(self.hooks, transition_gate=result)
            return ledger

    def bind_observed_training_reward(self, page):
        from .rewards import resolve, with_request
        gate = self.hooks.transition_gate
        if (not self._active or page is not self._page or gate.pending is not None or gate.stopped
                or page.observation.scene != 'reward_home' or page.observation.targets
                or tuple((page.reward or {}).get('frame_ids', ())) != page.frame_ids
                or not source_frames_match(page.observation, page.frame_ids)
                or validate_observation(page.observation) is not None
                or not 0 <= self.clock.now() - page.observation.revision <= 1000):
            raise ValueError('public-course reward settlement requires a fresh settled census destination')
        ledger, reward = resolve(self, page.reward, observation=page.observation)
        handoff = with_request(self.handoff, reward)
        handoff['nonbattle_context']['training_choice'] = dict(
            handoff['nonbattle_context']['training_choice'], status='complete')
        handoff = record_ledger_deck(handoff, self.ledger, ledger, frames=page.frame_ids, grants=reward.get('upgrades', ()))
        envelope = dict(schema_version=3, run_id=self.run_id,
            transition=transition_checkpoint(gate, self.clock.epoch), ledger=ledger.record(),
            ownership=self.ownership, handoff=handoff)
        write_hif_record(self.path, envelope)
        self.ledger, self.handoff = ledger, handoff

    def reconcile_submitted_training(self, gate, report, proof):
        """Read an already successful submission; retain its original failure."""
        from .rewards import request, confirm_page
        value = request(self)
        pending = gate.pending
        if (not self._active or value is None or value['stage'] != 'course_pending'
                or pending is None or pending.target.kind != 'confirm_training'
                or transition_checkpoint(gate, self.clock.epoch) != transition_checkpoint(self.hooks.transition_gate, self.clock.epoch)
                or report.get('route', {}).get('run_id') != self.run_id
                or report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid):
            raise ValueError('late public-course result does not belong to the original submitted transaction')
        events = [event for event in report['route'].get('input_events', ())
            if event.get('target_kind') == 'confirm_training' and event.get('target_key') == pending.target.key
            and event.get('source_revision') == pending.source.revision and event.get('issued_at_ms') == gate.issued_at_ms
            and event.get('dispatch_attempted') is True and event.get('success') is True and event.get('uncertain') is False]
        seen = proof.observation
        if (len(events) != 1 or seen.scene != 'training_result' or seen.run_id != self.run_id or seen.day != value['day']
                or validate_observation(seen) is not None
                or seen.revision <= pending.source.revision
                or not 0 <= seen.capture_age_ms + self.clock.now() - seen.revision <= 1000
                or not source_frames_match(seen, proof.frame_ids)
                or set(proof.frame_ids).intersection(frame.get('frame_id') for frame in report.get('frames', ()))):
            raise ValueError('late public-course result lacks a fresh independent complete result')
        old_handoff, old_page, old_confirmed = self.handoff, self._page, self._confirmed_reward
        original = transition_checkpoint(gate, self.clock.epoch)
        receipts = list((old_handoff or {}).get('recovery_navigation_receipts', ()))
        receipts.append(dict(outcome='late_verified_submitted_public_course_result', source_result=report['id'],
            original_transition=original, original_ledger_pending=self.ledger.record()['pending'],
            receipt_frame_ids=list(proof.frame_ids)))
        result = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            self._page = proof
            self._confirmed_reward = confirm_page(self, proof)
            self.handoff = dict(old_handoff or {}, recovery_navigation_receipts=receipts)
            self._save(transition_checkpoint(result, self.clock.epoch))
        except BaseException:
            self.handoff, self._page, self._confirmed_reward = old_handoff, old_page, old_confirmed
            raise
        return result

    def reconcile_reward_acknowledgement(self, gate, report, proof):
        """Confirm an already sent course navigation, keeping its reservation."""
        from .rewards import COURSE_RESULT_DESTINATIONS, request, confirm_page, retained_course_deck
        from .deletion_production import PICK_DESTINATIONS, SEARCH_DESTINATIONS
        value, pending = request(self), gate.pending
        if (not self._active or value is None or pending is None
                or (pending.target.kind, value['stage'], pending.source.scene) not in (
                    ('training_reward_ack', 'awaiting_ack', 'training_result'),
                    ('training_reward_upgrade_ack', 'upgrade', 'reward_upgrade'),
                    ('recovery_close', 'census', 'reward_inventory'),
                    ('reward_delete_select', 'deletion', 'reward_delete_choice'),
                    ('reward_delete_select', 'deletion', 'reward_delete_reselect'),
                    ('reward_delete_scroll', 'deletion', 'reward_delete_search'),
                    ('reward_delete_open', 'deletion', 'reward_delete_selected'),
                    ('reward_delete_confirm', 'deletion', 'reward_delete_confirm'),
                    ('reward_delete_result_ack', 'deletion', 'reward_delete_result'))
                or pending.expected_scenes != {
                    'training_reward_ack': COURSE_RESULT_DESTINATIONS,
                    'training_reward_upgrade_ack': COURSE_RESULT_DESTINATIONS,
                    'recovery_close': ('reward_home',),
                    'reward_delete_select': PICK_DESTINATIONS if pending.source.scene == 'reward_delete_choice' else ('reward_delete_selected',),
                    'reward_delete_scroll': SEARCH_DESTINATIONS,
                    'reward_delete_open': ('reward_delete_confirm', 'reward_delete_result'),
                    'reward_delete_confirm': ('reward_delete_result',),
                    'reward_delete_result_ack': ('reward_home',),
                }.get(pending.target.kind)
                or pending.target.kind == 'recovery_close' and (
                    pending.target.key != pending.target.kind or not value.get('inventory')
                    or value['inventory'].get('complete') is not True and not retained_course_deck(value, self.ledger.pending.before)
                    or value['inventory'].get('census_id') != value['transaction_id']
                    or value['inventory'].get('review_frame_ids')
                        != [row.get('frame_id') for row in report.get('frames', ())]
                    or value.get('navigation_source_frame_ids') != [row.get('frame_id') for row in report.get('frames', ())])
                or transition_checkpoint(gate, self.clock.epoch) != transition_checkpoint(self.hooks.transition_gate, self.clock.epoch)
                or report.get('route', {}).get('run_id') != self.run_id
                or report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid):
            raise ValueError('late reward destination does not belong to the original acknowledgement')
        events = [event for event in report['route'].get('input_events', ())
            if event.get('target_kind') == pending.target.kind and event.get('target_key') == pending.target.key
            and event.get('source_revision') == pending.source.revision and event.get('issued_at_ms') == gate.issued_at_ms
            and event.get('dispatch_attempted') is True and event.get('success') is True and event.get('uncertain') is False]
        seen = proof.observation
        course_ack = pending.target.kind in ('training_reward_ack', 'training_reward_upgrade_ack')
        detail = (proof.reward or {}).get('kind')
        interrupted = course_ack and detail == 'delete_choice'
        deleted = pending.target.kind in ('reward_delete_open', 'reward_delete_confirm') and (proof.reward or {}).get('kind') == 'delete_result'
        destination = ('reward_upgrade' if course_ack and detail == 'upgrade' else 'reward_delete_choice' if interrupted else
            'reward_delete_result' if deleted else
            'reward_' + detail if pending.target.kind == 'reward_delete_scroll' and detail in ('delete_choice', 'delete_search') else
            'reward_delete_reselect' if pending.target.kind == 'reward_delete_select' and detail == 'delete_reselect'
                and pending.source.scene == 'reward_delete_choice' else
                {'reward_delete_select': 'reward_delete_selected', 'reward_delete_open': 'reward_delete_confirm'}
                    .get(pending.target.kind, 'reward_home'))
        destination_day = value['next_day'] if destination == 'reward_home' else value['day']
        if (len(events) != 1 or seen.scene != destination or seen.run_id != self.run_id or seen.day != destination_day
                or validate_observation(seen) is not None
                or seen.revision <= pending.source.revision
                or not 0 <= seen.capture_age_ms + self.clock.now() - seen.revision <= 1000
                or not source_frames_match(seen, proof.frame_ids)
                or set(proof.frame_ids).intersection(frame.get('frame_id') for frame in report.get('frames', ()))):
            raise ValueError('late reward destination lacks the actual next day and independent fresh observations')
        original = transition_checkpoint(gate, self.clock.epoch)
        old_handoff, old_page, old_confirmed = self.handoff, self._page, self._confirmed_reward
        receipts = list((old_handoff or {}).get('recovery_navigation_receipts', ()))
        receipts.append(dict(outcome='late_verified_course_reward_destination', source_result=report['id'],
            original_transition=original, original_ledger_pending=self.ledger.record()['pending'],
            receipt_frame_ids=list(proof.frame_ids)))
        result = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            self._page = proof
            self._confirmed_reward = confirm_page(self, proof)
            self.handoff = dict(old_handoff or {}, recovery_navigation_receipts=receipts)
            self._save(transition_checkpoint(result, self.clock.epoch))
        except BaseException:
            self.handoff, self._page, self._confirmed_reward = old_handoff, old_page, old_confirmed
            raise
        return result

    def reconcile_reward_deletion_navigation(self, gate, report, proof):
        if gate.pending is None or not gate.pending.target.kind.startswith('reward_delete_'):
            raise ValueError('late course deletion lacks its original pending input')
        return self.reconcile_reward_acknowledgement(gate, report, proof)

    def reconcile_reward_inventory_navigation(self, gate, report, proof):
        """Finish the original read-only census navigation, keeping its cost."""
        from .rewards import request, confirm_page, optional_course_deck_review
        value, pending = request(self), gate.pending
        if (not self._active or value is None or value['stage'] != 'census' or pending is None
                or pending.target.kind not in ('recovery_inventory', 'recovery_inventory_detail', 'recovery_inventory_scroll')
                or transition_checkpoint(gate, self.clock.epoch) != transition_checkpoint(self.hooks.transition_gate, self.clock.epoch)
                or report.get('route', {}).get('run_id') != self.run_id
                or report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid):
            raise ValueError('late reward inventory navigation is not its original census')
        matching = [event for event in report['route'].get('input_events', ())
            if event.get('target_kind') == pending.target.kind and event.get('target_key') == pending.target.key
            and event.get('source_revision') == pending.source.revision and event.get('issued_at_ms') == gate.issued_at_ms
            and event.get('dispatch_attempted') is True and event.get('success') is True and event.get('uncertain') is False]
        inventory = (proof.reward or {}).get('inventory', {})
        old_inventory = value.get('inventory') or {}
        old = {row['index']: row for row in old_inventory.get('slots', ())}
        current = {row['index']: row for row in inventory.get('slots', ())}
        seen, registration = proof.observation, inventory.get('registration') or {}
        anchors = registration.get('anchor_indices', ())
        reviewed = optional_course_deck_review(value) and inventory.get('review_frame_ids') == list(proof.frame_ids)
        unfinished = reviewed and inventory.get('read_finished') is True and inventory.get('complete') is False
        if (len(matching) != 1 or seen.scene != 'reward_inventory' or seen.run_id != self.run_id or seen.day != value['next_day']
                or validate_observation(seen) is not None
                or seen.revision <= pending.source.revision or seen.page_id == pending.source.page_id
                or not 0 <= seen.capture_age_ms + self.clock.now() - seen.revision <= 1000
                or not source_frames_match(seen, proof.frame_ids)
                or set(proof.frame_ids).intersection(frame.get('frame_id') for frame in report.get('frames', ()))
                or inventory.get('census_id') != value['transaction_id']
                or not unfinished and inventory.get('detail_rejection') is not None):
            raise ValueError('late reward inventory lacks its independent original current census')
        if pending.target.kind == 'recovery_inventory':
            if (old_inventory or pending.source.scene != 'reward_home' or pending.source.day != value['next_day']
                    or pending.target.key != pending.target.kind or pending.expected_scenes != ('reward_inventory',)
                    or [row.get('frame_id') for row in report.get('frames', ())] != value.get('navigation_source_frame_ids')
                    or inventory.get('review_frame_ids') != list(proof.frame_ids)
                    or not (type(inventory.get('total')) is int and 1 <= inventory['total'] <= 99 or reviewed)):
                raise ValueError('late reward inventory opening is not its original first current census')
        elif unfinished:
            if (pending.source.scene != 'reward_inventory' or pending.source.day != value['next_day']
                    or pending.expected_scenes != ('reward_inventory',) or not old_inventory
                    or old_inventory.get('census_id') != value['transaction_id']
                    or value.get('navigation_source_frame_ids') != [row.get('frame_id') for row in report.get('frames', ())]):
                raise ValueError('incomplete course deck review differs from the original list navigation')
        elif (inventory.get('total') != old_inventory.get('total')
                or any(index not in current or current[index]['card_id'] != row['card_id']
                    or current[index]['presence'] != row['presence'] for index, row in old.items())
                or registration.get('source_frame_ids') != value.get('navigation_source_frame_ids')
                or registration.get('receipt_frame_ids') != list(proof.frame_ids)
                or registration.get('source_displacement') != old_inventory.get('displacement')
                or len(anchors) < 2 or any(index not in old for index in anchors)
                or len({old[index]['box'][0] for index in anchors}) < 2):
            raise ValueError('late reward inventory lacks independent original physical registration')
        if pending.target.kind == 'recovery_inventory_detail' and not unfinished:
            index = int(pending.target.key.split(':')[1])
            if (index not in old or current[index].get('detail', {}).get('frame_ids') != list(proof.frame_ids)
                    or current[index]['detail'].get('nonbattle_schema') != 1
                    or inventory.get('displacement') != old_inventory.get('displacement')
                    or registration.get('shifts') != [0] * len(proof.frame_ids)):
                raise ValueError('late reward detail is not its original unscrolled card instance')
        original = transition_checkpoint(gate, self.clock.epoch)
        old_handoff, old_page, old_confirmed = self.handoff, self._page, self._confirmed_reward
        receipts = list((old_handoff or {}).get('recovery_navigation_receipts', ()))
        receipts.append(dict(outcome='late_verified_reward_inventory_navigation', source_result=report['id'],
            original_transition=original, original_ledger_pending=self.ledger.record()['pending'],
            receipt_frame_ids=list(proof.frame_ids), registration=deepcopy(registration)))
        result = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            self._page = proof
            self._confirmed_reward = confirm_page(self, proof)
            self.handoff = dict(old_handoff or {}, recovery_navigation_receipts=receipts)
            self._save(transition_checkpoint(result, self.clock.epoch))
        except BaseException:
            self.handoff, self._page, self._confirmed_reward = old_handoff, old_page, old_confirmed
            raise
        return result

    def bind_observed_activity_reward(self, page):
        from .activity_reward import resolve, with_request
        gate = self.hooks.transition_gate
        if (not self._active or page is not self._page or gate.pending is not None or gate.stopped
                or page.observation.targets or not source_frames_match(page.observation, page.frame_ids)
                or not 0 <= page.observation.capture_age_ms + self.clock.now() - page.observation.revision <= 1000):
            raise ValueError('activity settlement lacks a fresh completed census destination')
        ledger, activity = resolve(self, page.activity)
        handoff = with_request(self.handoff, activity)
        handoff = record_ledger_deck(handoff, self.ledger, ledger, frames=page.frame_ids, grants=activity['cards'])
        envelope = dict(schema_version=3, run_id=self.run_id, transition=transition_checkpoint(gate, self.clock.epoch),
            ledger=ledger.record(), ownership=self.ownership, handoff=handoff)
        write_hif_record(self.path, envelope)
        self.ledger, self.handoff = ledger, handoff
        self._confirmed_activity = None

    def bind_observed_activity_transition(self, page):
        from .input_receipt import original_input_report
        from .activity_reward import frames, request, with_request
        gate, value, proof = self.hooks.transition_gate, request(self), page.activity
        if (not self._active or page is not self._page or value is None or value['kind'] != 'consult'
                or value['stage'] != 'consult_receipt' or gate.pending is None
                or gate.pending.target.kind != 'activity_ack_free_drink' or not isinstance(proof, dict)
                or proof.get('kind') != 'consult_reward_transition' or proof.get('day') != 6
                or tuple(frames(proof, (*value['source_frame_ids'], *value.get('navigation_source_frame_ids', ())))) != page.frame_ids
                or page.observation.targets or validate_observation(page.observation) is not None
                or not 0 <= page.observation.capture_age_ms + self.clock.now() - page.observation.revision <= 1000):
            raise ValueError('consult reward transition lacks its original fresh submitted acknowledgement')
        original_input_report(self, gate)
        quote, before = proof.get('quote') or {}, self.ledger.pending.before
        slots = proof.get('drink_slots') or ()
        if (type(quote.get('drink_rewards')) is not int or quote['drink_rewards'] != len(value['drinks']) + 1
                or type(quote.get('currency_delta')) is not int or quote['currency_delta'] < 0
                or proof.get('currency') != before.currency + quote['currency_delta']
                or (proof.get('stamina'), proof.get('max_stamina')) != (before.stamina, before.max_stamina)
                or len(slots) != 4 or sum(row.get('state') == 'OCCUPIED' for row in slots) != len(before.drinks) + quote['drink_rewards']):
            raise ValueError('consult support/P transition does not prove its actual quoted free rewards')
        old = value.get('consult_reward_quote')
        if old is not None:
            if old.get('quote') != quote: raise ValueError('consult original free support quote changed')
            return
        activity = deepcopy(value)
        activity['consult_reward_quote'] = dict(quote=deepcopy(quote), frame_ids=list(page.frame_ids),
            original_issued_at_ms=gate.issued_at_ms)
        handoff = with_request(self.handoff, activity)
        envelope = dict(schema_version=3, run_id=self.run_id, transition=transition_checkpoint(gate, self.clock.epoch),
            ledger=self.ledger.record(), ownership=self.ownership, handoff=handoff)
        write_hif_record(self.path, envelope)
        self.handoff = handoff

    def reconcile_activity_navigation(self, gate, report, proof):
        from .input_receipt import partial_first_lesson
        from .activity_reward import request, confirm_page, support_source
        pending, seen = gate.pending, proof.observation
        if (not self._active or pending is None or self.ledger is None or self.ledger.pending is None
                or transition_checkpoint(gate, self.clock.epoch) != transition_checkpoint(self.hooks.transition_gate, self.clock.epoch)
                or report.get('route', {}).get('run_id') != self.run_id
                or report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid):
            raise ValueError('late activity is not this original submitted navigation')
        support = support_source(pending) is not None and seen.scene == 'support_reward'
        matched = [event for event in report['route'].get('input_events', ())
            if event.get('target_kind') == pending.target.kind and event.get('target_key') == pending.target.key
            and event.get('source_revision') == pending.source.revision and event.get('issued_at_ms') == gate.issued_at_ms
            and event.get('dispatch_attempted') is True and (event.get('success') is True and event.get('uncertain') is False
                or support and partial_first_lesson(event, pending))]
        consult_support = (pending.target.kind in ('select_consult', 'confirm_consult')
            and seen.scene in ('support_reward', 'support_ok') and (request(self) or {}).get('kind') == 'consult')
        consult_deletion = (pending.target.kind == 'finish_consult' and seen.scene == 'reward_delete_choice'
            and (proof.reward or {}).get('kind') == 'delete_choice')
        if (len(matched) != 1 or seen.scene not in pending.expected_scenes and not (support or consult_support) or seen.run_id != self.run_id
                or validate_observation(seen) is not None or seen.revision <= pending.source.revision
                or not 0 <= seen.capture_age_ms + self.clock.now() - seen.revision <= 1000
                or not source_frames_match(seen, proof.frame_ids)
                or set(proof.frame_ids).intersection(frame.get('frame_id') for frame in report.get('frames', ()))):
            raise ValueError('late activity lacks its exact independent current destination')
        previous = self.handoff, self.ledger, self._page, self._confirmed_activity, self._confirmed_reward
        original_pending = self.ledger.record()['pending']
        receipts = list((self.handoff or {}).get('recovery_navigation_receipts', ()))
        receipts.append(dict(outcome='late_verified_activity_navigation', source_result=report['id'],
            original_transition=transition_checkpoint(gate, self.clock.epoch), original_ledger_pending=original_pending,
            receipt_frame_ids=list(proof.frame_ids)))
        result = TransitionGate(timeout_ms=gate.timeout_ms)
        try:
            self._page = proof
            if support:
                from .activity_reward import start_support_request
                self._confirmed_activity = start_support_request(self, proof)
            elif consult_deletion:
                from .rewards import consult_exit_request
                self._confirmed_reward = dict(consult_exit_request(self, gate), deletion=deepcopy(proof.reward['deletion']))
            elif request(self) is not None:
                self._confirmed_activity = confirm_page(self, proof)
            else:
                ledger = self.ledger.confirm(proof.receipt, transaction_id=self.ledger.pending.effect.transaction_id,
                    frame_ids=proof.receipt_frame_ids, observation=seen)
                self.handoff = record_ledger_deck(self.handoff, self.ledger, ledger, frames=proof.receipt_frame_ids)
                self.ledger = ledger
            self.handoff = dict(self.handoff or {}, recovery_navigation_receipts=receipts)
            self._save(transition_checkpoint(result, self.clock.epoch))
        except BaseException:
            self.handoff, self.ledger, self._page, self._confirmed_activity, self._confirmed_reward = previous
            raise
        return result

    def _save(self, record, *, create=False, cancel_unissued=False):
        gate = restore_transition(record, clock_epoch=self.clock.epoch)
        pending = gate.pending
        if pending is not None and pending.source.run_id != self.run_id:
            raise ValueError('HIF checkpoint run mismatch')
        candidate_ledger = self.ledger
        from . import activity_reward, customization_transaction
        from .rewards import request, with_request, start_request, permits_navigation
        course_navigation = candidate_ledger is not None and permits_navigation(self, candidate_ledger, gate)
        activity_navigation = candidate_ledger is not None and activity_reward.permits_navigation(self, candidate_ledger, gate)
        customization_navigation = candidate_ledger is not None and customization_transaction.permits_navigation(self, candidate_ledger, gate)
        reward_navigation = course_navigation or activity_navigation or customization_navigation
        reward = self._confirmed_reward if pending is None and not cancel_unissued else None
        activity = self._confirmed_activity if pending is None and not cancel_unissued else None
        customization = self._confirmed_customization if pending is None and not cancel_unissued else None
        if cancel_unissued and candidate_ledger is not None and candidate_ledger.pending is not None:
            reserved_reward = request(self)
            if reserved_reward is not None and reserved_reward['stage'] == 'course_pending':
                reward = dict(reserved_reward, stage='cancelled')
            reserved_activity = activity_reward.request(self)
            if reserved_activity is not None and reserved_activity['stage'] == 'paid_pending':
                activity = dict(reserved_activity, stage='cancelled')
            reserved_customization = customization_transaction.request(self)
            if reserved_customization is not None and reserved_customization['stage'] == 'awaiting_result':
                customization = dict(reserved_customization, stage='cancelled')
        saved_navigation = (self._navigation_only and self._active and self.ownership is not None
            and self.ownership['owner'] == 'task275' and self.ownership['phase'] == 'battle'
            and candidate_ledger is not None and candidate_ledger.pending is None
            and (pending is None or pending.target.kind in SAVED_CULTIVATION_NAVIGATION))
        if self._navigation_only and not saved_navigation:
            raise ValueError('saved-cultivation navigation cannot modify a resource or another pending action')
        recovery_navigation = saved_navigation or (self.ownership is not None and self.ownership['phase'] == 'recovery'
            and (pending is None or pending.target.kind in RECOVERY_NAVIGATION)
            and (candidate_ledger is None or candidate_ledger.pending is None))
        if self.ownership is not None and candidate_ledger is None and pending is not None:
            from .prestart import PRESTART_NAVIGATION
            from .start_transaction import START_AP_KIND, START_RESUME_CONTRACTS, START_RECEIPT_CONTRACTS
            recovery_navigation = recovery_navigation or (
                self.ownership['owner'] == 'task600' and self.ownership['phase'] == 'recovery'
                and pending.target.kind in (*PRESTART_NAVIGATION, START_AP_KIND, *START_RECEIPT_CONTRACTS, *START_RESUME_CONTRACTS)
                and not (self.handoff or {}).get('live_context')
                and not (self.handoff or {}).get('terminal_receipt'))
        if candidate_ledger is not None and not create and not recovery_navigation and not reward_navigation:
            if pending is not None and candidate_ledger.pending is None:
                page = self._page
                if page is None or page.observation != pending.source:
                    raise ValueError('HIF transaction requires a fresh complete page binding')
                if (page.baseline != candidate_ledger.state
                        or any(not isinstance(value, str) or not value for value in page.frame_ids)
                        or not ledger_frames_match(page.observation, page.frame_ids)
                        or pending.source.day != candidate_ledger.state.day):
                    raise ValueError('HIF transaction baseline was not independently confirmed')
                effects = tuple(effect for effect in page.effects if effect.target_key == pending.target.key)
                if len(effects) != 1:
                    raise ValueError('HIF action lacks a unique verified ledger effect')
                candidate_ledger = candidate_ledger.prepare(effects[0], frame_ids=page.frame_ids)
                if pending.target.kind == 'confirm_training':
                    reward = start_request(self, candidate_ledger, gate)
                elif pending.target.kind in ('execute_outing', 'confirm_consult') or (
                        pending.target.kind == 'select_consult' and candidate_ledger.pending.effect.phase == 'consult_rewards'):
                    activity = activity_reward.start_request(self, candidate_ledger, gate)
                elif (pending.target.kind == 'execute_customization' and self.ownership is not None
                        and page.customization_receipt is not None):
                    customization = customization_transaction.start_request(self, candidate_ledger, gate)
            elif pending is None and candidate_ledger.pending is not None:
                if cancel_unissued:
                    candidate_ledger = HifLedger(candidate_ledger.state,
                        card_catalog=self._catalog, drink_ids=self._drink_ids,
                        settled=candidate_ledger.settled)
                elif reward is not None or activity is not None or customization is not None:
                    pass
                elif self._confirmed_ledger is None:
                    raise ValueError('HIF ledger settlement not confirmed')
                else:
                    candidate_ledger = self._confirmed_ledger
        envelope = dict(schema_version=1, run_id=self.run_id, transition=record)
        if candidate_ledger is not None:
            envelope.update(schema_version=2, ledger=candidate_ledger.record())
        if self.ownership is not None:
            if self.ownership['phase'] == 'recovery' and pending is not None and not recovery_navigation:
                raise ValueError('HIF recovery cannot prepare a resource transaction')
            if candidate_ledger is None and (self.ownership['owner'] != 'task600'
                    or self.ownership['phase'] != 'recovery'):
                raise ValueError('HIF active input owner requires a complete ledger')
            handoff = self.handoff
            from .recovery_drinks import checkpoint as drink_checkpoint
            handoff = drink_checkpoint(self, gate, handoff, confirmed=self._confirmed_recovery_drinks,
                cancel_unissued=cancel_unissued)
            if reward is not None:
                handoff = with_request(handoff, reward)
            if activity is not None:
                handoff = activity_reward.with_request(handoff, activity)
            if customization is not None:
                handoff = customization_transaction.with_request(handoff, customization)
            if course_navigation and pending is not None and self.ledger.pending is not None:
                from .reward_production import navigation_checkpoint
                handoff = with_request(handoff, navigation_checkpoint(self, gate))
            if activity_navigation and pending is not None and self.ledger.pending is not None:
                handoff = activity_reward.with_request(handoff, activity_reward.navigation_checkpoint(self, gate))
            if customization_navigation and pending is not None and self.ledger.pending is not None:
                handoff = customization_transaction.with_request(handoff, customization_transaction.navigation_checkpoint(self, gate))
            from .deletion import deletion_checkpoint
            deletion = None if create or cancel_unissued else deletion_checkpoint(self, gate, candidate_ledger)
            if deletion is not None:
                handoff = deepcopy(self.handoff or {})
                history = deepcopy(handoff.get('nonbattle_context', {}))
                history['card_deletion'] = deletion
                handoff['nonbattle_context'] = history
            from .drink_replacement import replacement_checkpoint
            replacement = None if create or cancel_unissued else replacement_checkpoint(self, gate, candidate_ledger)
            if replacement is not None:
                handoff = deepcopy(handoff or {})
                history = deepcopy(handoff.get('nonbattle_context', {}))
                history['drink_replacement'] = replacement
                handoff['nonbattle_context'] = history
            if not create:
                from .start_transaction import start_checkpoint
                start = start_checkpoint(self, gate, cancel_unissued=cancel_unissued)
                if start is not None:
                    handoff = deepcopy(handoff or {})
                    handoff['cultivation_start'] = start
                from .closing_state import memory_checkpoint
                closing = memory_checkpoint(self, gate, cancel_unissued=cancel_unissued)
                if closing is not None:
                    handoff = dict(handoff or {})
                    handoff['closing'] = closing
            if saved_navigation:
                state = self._navigation_checkpoint(gate, cancel_unissued=cancel_unissued)
                if not isinstance(state, dict):
                    raise ValueError('saved-cultivation navigation metadata is invalid')
                # Only this child changes. Keep the unchanged multi-megabyte
                # budget/history private to the existing loaded record.
                handoff = dict(self.handoff)
                history = dict(handoff.get('nonbattle_context', {}))
                history['date_recovery'] = state
                handoff['nonbattle_context'] = history
            if candidate_ledger is not None and candidate_ledger is not self.ledger and candidate_ledger.pending is None:
                page = self._page
                frames = (page.receipt_frame_ids or page.frame_ids) if page is not None else ()
                handoff = record_ledger_deck(handoff, self.ledger, candidate_ledger, frames=frames)
            envelope.update(schema_version=3, ownership=self.ownership, handoff=handoff,
                ledger=candidate_ledger.record() if candidate_ledger is not None else None)
        write_hif_record(self.path, envelope, create=create)
        self.ledger = candidate_ledger
        if self.ownership is not None:
            self.handoff = handoff
        return True
