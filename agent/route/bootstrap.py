"""Bind the active Maa cultivation task and its local run state."""
from __future__ import annotations

import re
import json
import uuid
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass

from .hif import PRESET_ID, HifConfig
from .hif import SUPPORTED_IDOL as SUPPORTED_IDOL

_ENTRY = 'ProduceHifAdvanceEntry'
_ACTION = 'ProduceHifAdvance'
_RECOVERY_MEMO_NODES = frozenset('HifRecovery' + name for name in
    ('Notebook', 'Help', 'song', 'idol', 'focus'))
_NAVIGATION = frozenset({'recovery_menu', 'recovery_notebook', 'recovery_identity',
    'recovery_close', 'recovery_inventory', 'recovery_inventory_detail',
    'recovery_inventory_scroll', 'recovery_drink_detail', 'recovery_drink_close'})


def _product_paths():
    from utils.local_data import DEFAULT_RUNTIME_ROOT, PROJECT_ROOT

    runtime = DEFAULT_RUNTIME_ROOT / 'hif'
    return runtime / 'current-hif-host.json', runtime, PROJECT_ROOT / 'MaaGakumasu.exe'


class _RecordSnapshot:
    """A single owned parse, usable only inside its originating callback."""

    def __init__(self, raw, record, epoch):
        self._raw, self._record, self.epoch = raw, record, epoch
        self._active, self._consumed = True, False

    @property
    def raw_bytes(self):
        if not self._active:
            raise ValueError('hif_product_record_snapshot_expired')
        return self._raw

    def consume(self, current_bytes, epoch):
        if (not self._active or self._consumed or epoch != self.epoch
                or not isinstance(current_bytes, bytes)):
            raise ValueError('hif_product_record_snapshot_not_current')
        self._consumed = True
        record, self._record = self._record, None
        if current_bytes != self._raw:
            return None
        return record

    def close(self):
        self._active, self._record, self._raw = False, None, None


def _endpoint(context, *, expected_maa=None, controller_info=None, require_calibrated_size=True):
    """Read geometry/tokens only; every actual input still uses the Maa SDK."""
    import ctypes
    from ctypes import wintypes

    from utils.maa_input import verify_maa_input_endpoints

    info = context.tasker.controller.info if controller_info is None else controller_info
    if (not isinstance(info, dict) or info.get('type') != 'win32'
            or type(info.get('hwnd')) is not int or info['hwnd'] <= 0):
        raise ValueError('hif_product_controller_unresolved')
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    rect = wintypes.RECT()
    if not user32.GetClientRect(info['hwnd'], ctypes.byref(rect)):
        raise ValueError('hif_product_game_window_unavailable')
    result = verify_maa_input_endpoints(info, expected_client_size=(rect.right - rect.left, rect.bottom - rect.top),
                                        require_calibrated_size=require_calibrated_size)
    if (Path(result['maa']['executable']).resolve(strict=True) != (expected_maa or _product_paths()[2]).resolve(strict=True)
            or any(row.get('integrity_rid') != 0x3000 for row in result.values())):
        raise ValueError('hif_product_input_endpoint_changed')
    return result


def _process_instance(pid):
    """Normal Windows metadata only; failure cannot substitute a saved value."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
        wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise ValueError('hif_product_process_metadata_unavailable')
    try:
        path, length = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
        times = [wintypes.FILETIME() for _ in range(4)]
        if (not kernel.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(length))
                or not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times))):
            raise ValueError('hif_product_process_metadata_unavailable')
        created = times[0].dwHighDateTime << 32 | times[0].dwLowDateTime
        if not path.value or created <= 0:
            raise ValueError('hif_product_process_metadata_incomplete')
        return dict(executable=path.value, created_at_100ns=created)
    finally:
        kernel.CloseHandle(handle)


def _instance_endpoint(endpoint):
    result = deepcopy(endpoint)
    for key, row in result.items():
        instance = _process_instance(row['pid'])
        if ('executable' in row and Path(row['executable']) != Path(instance['executable'])
                or key == 'game' and Path(instance['executable']).name.lower() != 'gakumas.exe'):
            raise ValueError('hif_product_process_executable_changed')
        row.update(instance)
    return result


class _TaskScope:
    entry, node, action = 'Produce', _ENTRY, _ACTION

    def __init__(self, context, argv, path, *, factory=None):
        from maa.context import Context
        from maa.custom_action import CustomAction

        if (not isinstance(context, Context) or not isinstance(argv, CustomAction.RunArg)
                or argv.node_name != self.node or argv.custom_action_name != self.action
                or argv.task_detail.entry != self.entry
                or argv.task_detail.status.running is not True):
            raise ValueError('hif_product_not_native_produce_callback')
        # The native callback identity was checked above. Maa's remote
        # controller getter replaces its previous C++ proxy, so every consumer
        # in this scope must share this one borrowed controller instance.
        from .cultivation_factory import CultivationRecordBinding

        if type(factory) is not CultivationRecordBinding:
            raise ValueError('hif_product_factory_binding_invalid')
        self._factory = factory
        self.run_id = factory.run_id
        self.context = factory.task.context
        self.path = path
        self.task_id = argv.task_detail.task_id
        if type(self.task_id) is not int or self.task_id <= 0:
            raise ValueError('hif_product_task_identity_invalid')
        self.active, self.reason = True, None
        # Build the private decoded history before any fresh scene evidence.
        # Later dispatches still reread bytes and revalidate every live gate.
        self._pixel_record_decode = {}
        initial = self._read_record(pixel_decode=self._pixel_record_decode)
        self.game_pid = initial['ownership']['game_pid']
        self._endpoint_identity = None
        self._probe = None
        self._pin_instances = True
        self._require_navigation_budget = False
        self.probe_inputs = []
        self.permissions = frozenset(('capture', 'input', 'battle'))
        self._record_read_sink = None
        self._recovery_memo = self._recovery_model_binding = None

    def close(self):
        self.active = False
        if self._factory is not None:
            self._factory.close()
        self._pixel_record_decode.clear()
        if self._recovery_memo is not None:
            self._recovery_memo.close()
        if self._probe is not None:
            self._probe['navigation'] = None
            self._probe['dispatch'] = None

    def _read_record(self, **kwargs):
        return self._factory.read(self.path, **kwargs)

    def _require_current_task(self):
        if not self.active:
            raise ValueError('hif_product_permission_revoked')
        tasker = self.context.tasker
        job = self.context.get_task_job()
        detail = tasker.get_task_detail(self.task_id)
        if (not self.active or tasker.stopping or tasker.running is not True
                or job.job_id != self.task_id or job.status.running is not True
                or detail is None or detail.task_id != self.task_id
                or detail.entry != self.entry or detail.status.running is not True):
            raise ValueError('hif_product_task_stopped_or_changed')

    def _live(self, *, pixel_only=False, existing_pixels_only=False):
        self._require_current_task()
        if self._record_read_sink is not None:
            return self._read_record(read_sink=self._record_read_sink)
        if pixel_only:
            # The shared host is replaced atomically by its existing writers.
            # Revalidate a changed file, not the entire retained history for
            # each recognition of pixels already captured in this invocation.
            # Probes and new-run factory bindings keep their separate readers.
            if self._probe is None and (self._factory is None or existing_pixels_only):
                stat = self.path.stat()
                stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if self._pixel_record_decode.get('validated_file') == stamp:
                    return self._pixel_record_decode['record']
                record = self._read_record(pixel_decode=self._pixel_record_decode)
                current = self.path.stat()
                if stamp == (current.st_dev, current.st_ino, current.st_size,
                        current.st_mtime_ns, current.st_ctime_ns):
                    self._pixel_record_decode['validated_file'] = stamp
                return record
            return self._read_record(pixel_decode=self._pixel_record_decode)
        return self._read_record()

    def with_probe_record(self, kind, consumer):
        """Share this synchronous check's parse, never a previous phase's read."""
        if (self._probe is None or not self.active or self._record_read_sink is not None
                or kind not in ('capture', 'state') or not callable(consumer)):
            raise ValueError('hif_product_record_snapshot_scope_invalid')
        reads, snapshot = [], None
        self._record_read_sink = lambda raw, record: reads.append((raw, record))
        try:
            result = self.capture_endpoint() if kind == 'capture' else self.authorise_probe_state()
            if result is False:
                raise ValueError(self.reason or 'hif_product_record_snapshot_permission_rejected')
            if len(reads) != 1:
                raise ValueError('hif_product_record_snapshot_read_not_unique')
            raw, record = reads.pop()
            snapshot = _RecordSnapshot(raw, record, self._probe['clock'].epoch)
            result = consumer(result, snapshot)
            if not self.active or self._probe is None:
                raise ValueError('hif_product_record_snapshot_scope_revoked')
            return result
        except Exception:
            self.close()
            raise
        finally:
            self._record_read_sink = None
            if snapshot is not None:
                snapshot.close()

    def _prepare_recovery_memo(self, clock=None):
        """Bind read-only loaded resources once, before the first source frame.

        Only a verified native callback enables this path. Its loaded bundle remains
        fixed for this invocation; context node overrides invalidate the binding
        below. Unknown metadata keeps the original per-frame recognition path.
        """
        from importlib.metadata import PackageNotFoundError, version

        from live_decision.hif_recovery_memo import RecoveryCropMemo

        self.invalidate_recovery_memo()
        try:
            sdk = version('maafw')
            if sdk != '5.12.3':
                return
            resource = self.context.tasker.resource
            revision = resource.hash
            nodes = {node: self.context.get_node_data(node) for node in sorted(_RECOVERY_MEMO_NODES)}
            if not isinstance(revision, str) or not revision:
                return
            binding = dict(sdk=sdk, loaded_resource_revision=revision, nodes=nodes)
            json.dumps(binding, allow_nan=False)
        except (PackageNotFoundError, AttributeError, RuntimeError, TypeError, ValueError):
            return
        if clock is None:
            from .host import windows_route_clock
            clock = self._probe['clock'] if self._probe is not None else windows_route_clock()
        self._recovery_model_binding = binding
        self._recovery_memo = RecoveryCropMemo(call_id=f'{self.task_id}-{uuid.uuid4().hex}',
                                             epoch=clock.epoch)

    def invalidate_recovery_memo(self):
        if self._recovery_memo is not None:
            self._recovery_memo.close()
        self._recovery_memo = self._recovery_model_binding = None

    def _read_endpoint(self, *, controller_info=None):
        expected_maa = self._factory.task.expected_maa if self._factory is not None else None
        return _endpoint(self.context, controller_info=controller_info, expected_maa=expected_maa)

    def _observe_endpoint(self):
        info = self.context.tasker.controller.info
        return self._read_endpoint(controller_info=info), info['hwnd']

    def _identity(self, result, hwnd):
        return (hwnd, tuple(result['game']['observed_client_size']),
            tuple(sorted((key, row['pid'], row.get('executable'), row.get('created_at_100ns'))
                for key, row in result.items())))

    def _bound_record(self, *, pixel_only=False, existing_pixels_only=False):
        record = self._live(pixel_only=True, existing_pixels_only=existing_pixels_only) if pixel_only else self._live()
        saved_pid = self.game_pid
        if self._probe is not None:
            original = self._probe['original']
            if any(record[key] != original[key] for key in ('run_id', 'ownership', 'transition', 'ledger')):
                raise ValueError('hif_product_probe_business_changed')
            saved_pid = original['ownership']['game_pid']
        if record['ownership']['game_pid'] != saved_pid:
            raise ValueError('hif_product_input_endpoint_changed')
        return record

    def endpoint(self):
        return self._verify_current_endpoint(self._bound_record(pixel_only=True))

    def _verify_current_endpoint(self, record):
        """Consume this call's fresh record; the OS endpoints are never reused."""
        result, hwnd = self._observe_endpoint()
        if self._pin_instances:
            result = _instance_endpoint(result)
        if result['game']['pid'] != self.game_pid:
            raise ValueError('hif_product_input_endpoint_changed')
        if self._factory is not None:
            self._factory.task.current()
        identity = self._identity(result, hwnd)
        if self._endpoint_identity is not None and identity != self._endpoint_identity:
            raise ValueError('hif_product_input_process_changed')
        self._endpoint_identity = identity
        return result

    def begin_process_probe(self, clock):
        try:
            return self._begin_process_probe(clock)
        except Exception as error:
            self.reason = str(error)
            self.close()
            raise

    def _begin_process_probe(self, clock):
        """Internal preparation only; no caller-configurable input grant."""
        record = self._live()
        owner = record['ownership']
        if (self._endpoint_identity is not None or self._probe is not None
                or owner['owner'] != 'task600' or owner['phase'] != 'nonbattle'
                or record['transition']['pending'] is not None or record['ledger']['pending'] is not None
                or record['transition'].get('clock_epoch') != clock.epoch):
            raise ValueError('hif_product_process_probe_scope_invalid')
        endpoint = self._read_endpoint()
        pid = endpoint['game']['pid']
        if type(pid) is not int or pid <= 0 or pid == self.game_pid:
            raise ValueError('hif_product_process_probe_needs_new_endpoint')
        self.game_pid = pid
        self._pin_instances = True
        self.permissions = frozenset(('capture',))
        self._probe = dict(original=deepcopy(record), clock=clock, navigation=None, dispatch=None, consumed=set())
        self.endpoint()
        from live_decision.hif_capture_adapter import adapt_hif_context

        # Guard the canonical action before the existing adapter maps it once
        # to Maa's slightly rounded 718..722px capture coordinates.
        return _Context(adapt_hif_context(self.context, input_guard=self._dispatch_navigation), self)

    def approve_navigation(self, intent, *, budget_guard=None, snapshot_budget_guard=None):
        """Reserve one fresh, exact free-navigation action from the producer."""
        try:
            if self._probe is None or self._probe['navigation'] is not None or self._probe['dispatch'] is not None:
                raise ValueError('hif_product_navigation_not_available')
            if (budget_guard is not None and not callable(budget_guard)
                    or snapshot_budget_guard is not None and not callable(snapshot_budget_guard)
                    or budget_guard is not None and snapshot_budget_guard is not None
                    or self._require_navigation_budget and budget_guard is None and snapshot_budget_guard is None):
                raise ValueError('hif_product_navigation_budget_guard_missing')
            self.endpoint()
            action, frames = intent.get('action'), intent.get('frame_ids')
            kind = intent.get('target_kind')
            if (intent.get('entry') != 'ProduceHifAdvanceClick' or kind not in _NAVIGATION
                    or type(intent.get('source_revision')) is not int or intent['source_revision'] < 1
                    or type(intent.get('issued_at_ms')) is not int
                    or not isinstance(frames, (list, tuple)) or len(frames) != 1 or not isinstance(frames[0], str)
                    or not isinstance(action, dict)):
                raise ValueError('hif_product_navigation_intent_invalid')
            from .maa_bridge import route_input_action

            box = action.get('target')
            if kind == 'recovery_inventory_scroll':
                begin, end = action.get('begin'), action.get('end')
                if (not all(isinstance(value, list) and len(value) == 4
                        and all(type(item) is int for item in value) for value in (begin, end))
                        or begin[0] != end[0] or begin[2:] != [1, 1] or end[2:] != [1, 1]):
                    raise ValueError('hif_product_navigation_geometry_invalid')
                box = [end[0], end[1], 1, begin[1] - end[1]]
            if (not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box)
                    or min(box[:2]) < 0 or min(box[2:]) <= 0
                    or box[0] + box[2] > 720 or box[1] + box[3] > 1280
                    or action != route_input_action(kind, box)):
                raise ValueError('hif_product_navigation_geometry_invalid')
            match = re.fullmatch(r'route-frame-' + str(self.game_pid) + r'-([0-9]+)', frames[0])
            if match is None:
                raise ValueError('hif_product_navigation_frame_process_changed')
            captured_at = int(match[1])
            now = self._probe['clock'].now()
            if not 0 <= now - captured_at <= 1000 or not captured_at <= intent['issued_at_ms'] <= now:
                raise ValueError('hif_product_navigation_evidence_expired')
            key = (intent['source_revision'], tuple(frames), kind)
            if key in self._probe['consumed']:
                raise ValueError('hif_product_navigation_already_dispatched')
            self._probe['navigation'] = dict(intent=deepcopy(intent), key=key,
                captured_at_ms=captured_at, budget_guard=budget_guard,
                snapshot_budget_guard=snapshot_budget_guard)
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
            self.reason = str(error)
            self.close()
            return False

    def consume_navigation(self, args, kwargs):
        try:
            if self._probe is None or self._probe['navigation'] is None:
                raise ValueError('hif_product_navigation_not_approved')
            ticket, self._probe['navigation'] = self._probe['navigation'], None
            # Consume before checking/dispatch so an exception never reissues it.
            self._probe['consumed'].add(ticket['key'])
            intent = ticket['intent']
            if args != (intent['entry'],) or kwargs != {'pipeline_override': {intent['entry']: intent['action']}}:
                raise ValueError('hif_product_navigation_dispatch_changed')
            self._probe['dispatch'] = ticket
        except Exception as error:
            self.reason = str(error)
            self.close()
            raise

    def _dispatch_navigation(self):
        """The adapter calls this after mapping, immediately before Maa input."""
        from .maa_bridge import RouteInputNotDispatched

        try:
            if self._probe is None or self._probe['dispatch'] is None:
                raise ValueError('hif_product_navigation_not_prepared')
            ticket, self._probe['dispatch'] = self._probe['dispatch'], None
            if ticket.get('snapshot_budget_guard') is not None:
                if self.with_probe_record('capture', ticket['snapshot_budget_guard']) is not True:
                    raise ValueError('hif_product_navigation_budget_rejected')
            else:
                endpoints = self.capture_endpoint()
                if ticket['budget_guard'] is not None and ticket['budget_guard'](endpoints) is not True:
                    raise ValueError('hif_product_navigation_budget_rejected')
            if not 0 <= self._probe['clock'].now() - ticket['captured_at_ms'] <= 1000:
                raise ValueError('hif_product_navigation_evidence_expired')
            self.probe_inputs.append(dict(intent=deepcopy(ticket['intent']), dispatch_attempted=True,
                success=None, uncertain=True))
            return True
        except Exception as error:
            self.reason = str(error)
            self.close()
            raise RouteInputNotDispatched(str(error)) from error

    def finish_process_probe(self, receipt):
        """Promote only a newly committed proof, never the producer's return alone."""
        from .process_resume import read_bound_game_pid

        try:
            if (self._probe is None or self._probe['navigation'] is not None
                    or self._probe['dispatch'] is not None):
                raise ValueError('hif_product_process_probe_not_complete')
            original = self._probe['original']
            record = self._live()
            endpoints, hwnd = self._observe_endpoint()
            endpoints = _instance_endpoint(endpoints)
            pid = read_bound_game_pid(record, run_id=self.run_id,
                original_game_pid=original['ownership']['game_pid'], current_endpoints=endpoints)
            if (not isinstance(receipt, dict) or receipt.get('changed') is not True
                    or receipt.get('game_pid') != pid or pid != self.game_pid
                    or receipt.get('binding_revision') != record['ownership']['revision']
                    or record['ownership'] != dict(original['ownership'], game_pid=pid,
                        revision=original['ownership']['revision'] + 1)
                    or any(record[key] != original[key] for key in ('ledger', 'transition'))
                    or self._identity(endpoints, hwnd) != self._endpoint_identity):
                raise ValueError('hif_product_process_publication_changed')
            self._probe = None
            self.endpoint()
            self.permissions = frozenset(('capture', 'input', 'battle'))
        except Exception as error:
            self.reason = str(error)
            self.close()
            raise

    def permits(self, kind):
        if not self.active or kind not in self.permissions:
            return False
        try:
            self.capture_endpoint() if kind == 'capture' else self.endpoint()
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
            self.reason = str(error)
            self.close()
            return False

    def authorise_capture(self):
        return self.permits('capture')

    def capture_endpoint(self):
        """One fresh native-task/capture grant and endpoint check per call."""
        try:
            if not self.active or 'capture' not in self.permissions:
                raise ValueError('hif_product_capture_not_authorised')
            if (self._factory is not None and self._probe is None
                    and self._endpoint_identity is not None):
                # The factory and original game were bound at startup. A new
                # screenshot needs the current task and actual endpoints, not
                # permission to spend the retained execution budget. This is
                # a fresh check, not a cached grant. Dispatch still reads and
                # validates the complete current record through require().
                self._require_current_task()
                return self._verify_current_endpoint(None)
            return self.endpoint()
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
            self.reason = str(error)
            self.close()
            raise

    def require_pixel_recognition(self):
        """Authorize existing pixels only; no capture or input endpoint grant."""
        try:
            if not self.active or 'capture' not in self.permissions:
                raise ValueError('hif_product_capture_not_authorised')
            # Recognizing an existing image cannot send input or capture a new
            # frame. Reuse only its unchanged-file view, as the legacy binding
            # already does. New captures separately check the live task and
            # actual endpoints. Dispatch still reads and validates the full
            # current record, cumulative budget, task and endpoints every time.
            self._bound_record(pixel_only=True, existing_pixels_only=True)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
            self.reason = str(error)
            self.close()
            raise

    def authorise_probe_state(self):
        """Permit only current private probe bookkeeping, never SDK I/O."""
        if self._probe is None:
            return False
        try:
            self.require_pixel_recognition()
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
            return False

    def authorise_input(self):
        return self.permits('input')

    def authorise_route_preparation(self):
        """Check current route authority without an intermediate OS input gate.

        Preparation cannot send input. The bound Context still checks the live
        endpoints and exact durable intent immediately before the Maa call.
        """
        if (not self.active or 'input' not in self.permissions or self._probe is not None
                or self._endpoint_identity is None):
            return False
        try:
            if self._bound_record()['ownership']['owner'] != 'task600':
                raise ValueError('hif_product_route_preparation_owner_changed')
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
            self.reason = str(error)
            self.close()
            return False

    def authorise_battle(self):
        return self.permits('battle')

    def authorise_battle_navigation_preparation(self):
        """Check the current battle navigation scope, without granting SDK I/O.

        The date producer retains its business/budget checks. Capture and the
        final dispatch still verify the current native endpoints separately.
        """
        if (not self.active or not {'capture', 'input', 'battle'} <= self.permissions
                or self._probe is not None or self._endpoint_identity is None):
            return False
        try:
            owner = self._bound_record()['ownership']
            if (owner['owner'], owner['phase']) != ('task275', 'battle'):
                raise ValueError('hif_product_battle_navigation_preparation_owner_changed')
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
            self.reason = str(error)
            self.close()
            return False

    def require(self, kind):
        if kind == 'dispatch':
            if self._probe is not None:
                self.close()
                raise ValueError('hif_product_probe_requires_exact_navigation')
            # The existing owner lease remains authoritative for the action.
            # This additional gate revokes the whole product task, including
            # a battle already inside run_hif_battle, when Maa is stopped.
            try:
                record = self._bound_record(pixel_only=True)
                kind = 'battle' if record['ownership']['owner'] == 'task275' else 'input'
                if kind not in self.permissions:
                    raise ValueError('hif_product_permission_revoked')
                endpoints = self._verify_current_endpoint(record)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
                self.reason = str(error)
                self.close()
                raise ValueError(self.reason) from error
            return endpoints
        if not self.permits(kind):
            raise ValueError(self.reason or 'hif_product_permission_revoked')


class _Proxy:
    def __init__(self, wrapped, scope):
        object.__setattr__(self, '_product_wrapped', wrapped)
        object.__setattr__(self, '_product_scope', scope)

    def __getattr__(self, name):
        if name.startswith('post_'):
            raise ValueError('hif_product_unapproved_sdk_operation')
        return getattr(self._product_wrapped, name)

    def __setattr__(self, name, value):
        if name.startswith('_product_'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._product_wrapped, name, value)


class _PinnedTasker(_Proxy):
    def __init__(self, wrapped):
        super().__init__(wrapped, None)
        self._product_controller = wrapped.controller

    @property
    def controller(self):
        return self._product_controller


class _PinnedContext(_Proxy):
    """Keep borrowed SDK objects stable, without caching any live evidence."""

    def __init__(self, wrapped):
        super().__init__(wrapped, None)
        self._product_tasker = _PinnedTasker(wrapped.tasker)

    @property
    def tasker(self):
        return self._product_tasker


class _Controller(_Proxy):
    def capture_with_record_guard(self, guard, *args, **kwargs):
        try:
            if not callable(guard) or self._product_scope.with_probe_record('capture',
                    lambda native, snapshot: guard(capture_endpoints=native, record_snapshot=snapshot)) is not True:
                raise ValueError('hif_product_capture_budget_rejected')
        except Exception as error:
            self._product_scope.reason = str(error)
            self._product_scope.close()
            raise
        return self._product_wrapped.post_screencap(*args, **kwargs)

    def capture_with_guard(self, guard, *args, **kwargs):
        try:
            if not callable(guard) or guard(self._product_scope.capture_endpoint()) is not True:
                raise ValueError('hif_product_capture_budget_rejected')
        except Exception as error:
            self._product_scope.reason = str(error)
            self._product_scope.close()
            raise
        return self._product_wrapped.post_screencap(*args, **kwargs)

    def post_screencap(self, *args, **kwargs):
        self._product_scope.require('capture')
        return self._product_wrapped.post_screencap(*args, **kwargs)

    @property
    def cached_image(self):
        self._product_scope.require_pixel_recognition()
        return self._product_wrapped.cached_image

    def post_click(self, *args, **kwargs):
        self._product_scope.require('dispatch')
        return self._product_wrapped.post_click(*args, **kwargs)

    def post_click_with_guard(self, x, y, *, final_guard):
        """Run the final scene guard after all task/record/endpoint work."""
        if not callable(final_guard):
            raise ValueError('hif_product_final_input_guard_not_callable')
        self._product_scope.require('dispatch')
        if final_guard() is not True:
            raise ValueError('hif_product_final_input_guard_rejected')
        return self._product_wrapped.post_click(x, y)

    def post_click_with_endpoint_guard(self, x, y, *, final_guard):
        """Share this dispatch's verified endpoints with its final scene guard."""
        if not callable(final_guard):
            raise ValueError('hif_product_final_input_guard_not_callable')
        endpoints = self._product_scope.require('dispatch')
        if final_guard(endpoints) is not True:
            raise ValueError('hif_product_final_input_guard_rejected')
        return self._product_wrapped.post_click(x, y)

    def post_swipe(self, *args, **kwargs):
        self._product_scope.require('dispatch')
        return self._product_wrapped.post_swipe(*args, **kwargs)


class _Tasker(_Proxy):
    def __init__(self, wrapped, scope):
        super().__init__(wrapped, scope)
        self._product_controller = _Controller(wrapped.controller, scope)

    @property
    def controller(self):
        return self._product_controller


class _Context(_Proxy):
    def __init__(self, wrapped, scope):
        super().__init__(wrapped, scope)
        self._product_tasker = _Tasker(wrapped.tasker, scope)

    @property
    def tasker(self):
        return self._product_tasker

    def _require_recognition(self, args, kwargs, image_index):
        import numpy as np

        image = kwargs.get('image', args[image_index] if len(args) > image_index else None)
        if (isinstance(image, np.ndarray) and image.dtype == np.uint8
                and image.ndim == 3 and image.shape[2] == 3 and image.size):
            self._product_scope.require_pixel_recognition()
        else:
            self._product_scope.require('capture')

    def run_recognition(self, *args, **kwargs):
        self._require_recognition(args, kwargs, 1)
        overrides = kwargs.get('pipeline_override', args[2] if len(args) > 2 else None)
        if isinstance(overrides, dict) and any(key in overrides for key in _RECOVERY_MEMO_NODES):
            self._product_scope.invalidate_recovery_memo()
        try:
            return self._product_wrapped.run_recognition(*args, **kwargs)
        except Exception:
            self._product_scope.invalidate_recovery_memo()
            self._product_scope._pixel_record_decode.clear()
            raise

    def override_pipeline(self, *args, **kwargs):
        self._product_scope.invalidate_recovery_memo()
        return self._product_wrapped.override_pipeline(*args, **kwargs)

    def recovery_crop_recognition(self, image, field, parameters, *, frame_id, epoch, roi=None, capture_controller=None):
        try:
            return self._recovery_crop_recognition(image, field, parameters, frame_id=frame_id,
                epoch=epoch, roi=roi, capture_controller=capture_controller)
        except Exception:
            self._product_scope.invalidate_recovery_memo()
            self._product_scope._pixel_record_decode.clear()
            raise

    def _recovery_crop_recognition(self, image, field, parameters, *, frame_id, epoch, roi, capture_controller):
        from live_decision.hif_recovery_memo import recovery_field_spec
        from live_decision.hif_capture_adapter import _existing_adapter

        scope = self._product_scope
        if not scope.active:
            raise ValueError(scope.reason or 'hif_product_permission_revoked')
        roi, _ = recovery_field_spec(field, roi)
        x, y, w, h = roi
        node = 'HifRecovery' + field
        recognize = lambda crop: self._product_wrapped.run_recognition(node, crop,
            pipeline_override={node: parameters})
        def guard():
            scope.require_pixel_recognition()
            return True
        if scope._recovery_memo is None:
            guard()
            return recognize(image[y:y+h, x:x+w].copy()), None
        if (epoch != scope._recovery_memo.epoch
                or re.fullmatch(r'route-frame-' + str(scope.game_pid) + r'-[0-9]+', frame_id) is None):
            raise ValueError('hif_product_recovery_crop_frame_scope_changed')
        # The existing adapter proves that these are pixels from this context;
        # old pixel handles never become a new input grant or a new timestamp.
        original = self._product_wrapped.tasker.controller
        adapter = _existing_adapter(original if capture_controller is None else capture_controller)
        if (adapter is None or adapter is not _existing_adapter(original) and not (
                getattr(adapter._hif_wrapped, '_product_scope', None) is scope
                and adapter._hif_wrapped._product_wrapped is scope.context.tasker.controller)):
            raise ValueError('hif_product_recovery_crop_capture_scope_changed')
        metadata = adapter.frame_metadata(image)
        model = dict(binding=scope._recovery_model_binding, raw_shape=metadata['raw_shape'],
                     effective_shape=metadata['effective_shape'])
        return scope._recovery_memo.recognise(field, image, call_id=scope._recovery_memo.call_id,
            epoch=epoch, frame_id=frame_id, model_key=model, parameters=parameters,
            guard=guard, recognise=recognize, roi=roi)

    def run_recognition_direct(self, *args, **kwargs):
        self._require_recognition(args, kwargs, 2)
        try:
            return self._product_wrapped.run_recognition_direct(*args, **kwargs)
        except Exception:
            self._product_scope.invalidate_recovery_memo()
            self._product_scope._pixel_record_decode.clear()
            raise

    def run_action(self, *args, **kwargs):
        if self._product_scope._probe is not None:
            count = len(self._product_scope.probe_inputs)
            try:
                self._product_scope.consume_navigation(args, kwargs)
                result = self._product_wrapped.run_action(*args, **kwargs)
                if len(self._product_scope.probe_inputs) == count + 1:
                    row = self._product_scope.probe_inputs[-1]
                    row['success'] = getattr(result, 'success', None) is True
                    row['uncertain'] = not row['success']
                return result
            except Exception:
                self._product_scope.close()
                raise
        entry = args[0] if args else kwargs.get('entry')
        if entry == 'ProduceHifAdvanceClick':
            from .maa_bridge import RouteInputNotDispatched

            try:
                self._product_scope.require('dispatch')
                host = getattr(self._product_wrapped, 'gkh_hif_route_host', None)
                validate = getattr(host, 'validate_dispatch', None)
                if not callable(validate) or validate(args, kwargs) is not True:
                    raise ValueError('hif_product_route_dispatch_not_validated')
            except Exception as error:
                self._product_scope.reason = str(error)
                self._product_scope.close()
                raise RouteInputNotDispatched(str(error)) from error
        else:
            self._product_scope.require('dispatch')
        return self._product_wrapped.run_action(*args, **kwargs)

    def run_task(self, *args, **kwargs):
        raise ValueError('hif_product_unapproved_sdk_operation')

    run_action_direct = run_task


@dataclass
class HifBootstrapBinding:
    context: object
    runtime: object
    scope: _TaskScope

    def close(self):
        self.scope.close()


def _assemble(scope, runtime_path, *, evidence_sink=None):
    from .ownership import locked_hif_record
    from .production import bind_hif_nonbattle_runtime
    from .cultivation import HifCultivationRuntime

    if scope._probe is not None or not {'capture', 'input', 'battle'} <= scope.permissions:
        raise ValueError('hif_product_process_probe_not_complete')
    endpoint = scope.endpoint()
    if scope._recovery_memo is None:
        scope._prepare_recovery_memo()
    context = _Context(scope.context, scope)
    # The scope fixes the retained legacy run or the native factory's UUID.
    # Assembly itself never rotates a run or reconstructs its old budget.
    with locked_hif_record(scope.path):
        scope._bound_record()
        bound = bind_hif_nonbattle_runtime(context, path=scope.path, run_id=scope.run_id, game_pid=scope.game_pid,
            client_size=tuple(endpoint['game']['observed_client_size']),
            authorise_capture=scope.authorise_capture, authorise_input=scope.authorise_input,
            authorise_preparation=scope.authorise_route_preparation, evidence_sink=evidence_sink)
    runtime = HifCultivationRuntime(context=context, path=scope.path, run_id=scope.run_id, game_pid=scope.game_pid,
        runtime=runtime_path, authorise_capture=scope.authorise_capture,
        authorise_input=scope.authorise_input, authorise_battle=scope.authorise_battle, endpoint=scope.endpoint,
        authorise_preparation=scope.authorise_route_preparation,
        authorise_navigation_preparation=scope.authorise_battle_navigation_preparation,
        evidence_sink=evidence_sink)
    runtime.bound = bound
    bound.gkh_hif_cultivation_runtime = runtime
    return HifBootstrapBinding(bound, runtime, scope)


def bind_hif_product_resume(context, argv, config):
    """Start or resume from the product's real native Produce callback."""
    config = HifConfig.from_mapping(vars(config))
    if config.mode != 'advance' or config.preset != PRESET_ID:
        return None
    if not hasattr(argv, 'task_detail') or not callable(getattr(context, 'get_task_job', None)):
        return None
    path, runtime, maa = _product_paths()
    try:
        raw = json.loads(path.read_bytes())
    except FileNotFoundError:
        return _bind_product_factory(context, argv, config, path, runtime, None, expected_maa=maa)
    if not isinstance(raw, dict):
        raise ValueError('HIF product run state is invalid')
    return _bind_product_factory(context, argv, config, path, runtime, raw, expected_maa=maa)


def _bind_product_factory(context, argv, config, path, runtime, record, *, expected_maa=None):
    from . import cultivation_factory
    from .host import windows_route_clock
    from .new_run import start_record
    from .battle_policy import FAST_PROFILE, UINET21_PROFILE, retained_profile, selected_profile
    task, scope = None, None
    try:
        task = cultivation_factory.NativeCultivationTask(context, argv, runtime=runtime,
            expected_maa=expected_maa or _product_paths()[2], entry='Produce')
        clock = windows_route_clock()
        previous_start = start_record(record.get('handoff')) if record is not None else None
        policy_preflight = None
        if config.battle_policy == 'strong-reference-trial' and previous_start is None:
            from .battle_policy import preflight_strong_provider
            # Resource/configuration failures happen before archive/rotation
            # or AP. Constructor checks do not start a search or prove quality.
            policy_preflight = preflight_strong_provider()
        else:
            # Factories without completed-task allocation still resume the
            # existing cultivation. They must not require a new-run preflight.
            next_run = getattr(cultivation_factory, 'completed_native_task_for_next_run', None)
            if previous_start is not None and previous_start.get('status') == 'available':
                network_profile = retained_profile(record['handoff'], selection=config.battle_policy)
            elif previous_start is None or (previous_start.get('status') == 'complete'
                    and next_run is not None and next_run(record, task.current())):
                network_profile = selected_profile(config.battle_policy)
            else:
                network_profile = None
            if network_profile in (FAST_PROFILE, UINET21_PROFILE):
                from live_decision.hif_gkms_runtime import require_resources, battle_identity_master, require_uinet21_resources

                (require_resources if network_profile == FAST_PROFILE else require_uinet21_resources)()
                battle_identity_master()
        terminal = (cultivation_factory.completed_terminal_report(record, runtime)
            if record is not None and previous_start is None else None)
        plan = cultivation_factory.bind_cultivation_task(
            path, task=task, config=config, terminal_report=terminal, clock=clock)
        if plan.observe_only or plan.needs_same_save_proof:
            raise ValueError('hif_product_current_save_proof_required')
        factory = cultivation_factory.CultivationRecordBinding(task, plan, path=path, clock=clock)
        scope = _TaskScope(context, argv, path, factory=factory)
        scope.policy_preflight = policy_preflight
        scope.customization_configuration = deepcopy(
            (start_record(json.loads(path.read_bytes()).get('handoff')) or {}).get('customization_selection'))
        if not all(callback() for callback in (
                scope.authorise_capture, scope.authorise_input, scope.authorise_battle)):
            raise ValueError(scope.reason or 'hif_product_cultivation_not_authorised')
        binding = _assemble(scope, runtime)
        if policy_preflight is not None:
            binding.context.gkh_hif_policy_preflight = deepcopy(policy_preflight)
        return binding
    except Exception:
        if scope is not None:
            scope.close()
        elif task is not None:
            task.close()
        raise
