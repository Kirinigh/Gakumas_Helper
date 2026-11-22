"""Opt-in three-frame CPU OCR. Children receive pixels, never a game controller.

The caller owns capture permissions, source age, page validation and input. This
pool neither captures nor binds a controller, and is not enabled by any entry.
"""
import os
import sys
import json
import time
import uuid
import queue
import struct
import hashlib
import threading
import subprocess
import unicodedata
from pathlib import Path

import numpy as np

VERSION = 'hif-full-pixel-cpu-v1'
WORKERS = 3
OCR = dict(expected=['.+'], roi=[0, 0, 720, 1280])
_MAX_PACKET = 32 * 1024 * 1024


class PixelOcrError(RuntimeError):
    """No partial batch or previous result may replace a failed observation."""


def _write_packet(stream, message, images=()):
    buffers = [np.ascontiguousarray(image).tobytes() for image in images]
    descriptors = [dict(shape=list(image.shape), size=len(data))
                   for image, data in zip(images, buffers, strict=True)]
    header = json.dumps(dict(message=message, images=descriptors),
                        ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(header) + sum(map(len, buffers)) > _MAX_PACKET:
        raise PixelOcrError('pixel packet too large')
    stream.write(struct.pack('!I', len(header)))
    stream.write(header)
    for data in buffers:
        stream.write(data)
    stream.flush()


def _read_exact(stream, size):
    parts = bytearray()
    while len(parts) < size:
        part = stream.read(size - len(parts))
        if not part:
            raise EOFError('pixel worker pipe closed')
        parts.extend(part)
    return parts


def _read_packet(stream):
    length, = struct.unpack('!I', _read_exact(stream, 4))
    if not 0 < length <= _MAX_PACKET:
        raise PixelOcrError('invalid pixel header size')
    packet = json.loads(_read_exact(stream, length))
    images, total = [], length
    for item in packet['images']:
        shape, size = item['shape'], item['size']
        if (len(shape) != 3 or any(type(n) is not int or n < 0 for n in shape)
                or shape[2] != 3 or type(size) is not int
                or size != shape[0] * shape[1] * shape[2]):
            raise PixelOcrError('invalid pixel shape')
        total += size
        if total > _MAX_PACKET:
            raise PixelOcrError('invalid pixel payload size')
        image = np.frombuffer(_read_exact(stream, size), dtype=np.uint8).reshape(shape)
        image.flags.writeable = False
        images.append(image)
    return packet['message'], images


def _rows(detail):
    return [dict(text=''.join(unicodedata.normalize('NFKC', row['text']).split()),
                 box=list(row['box']), score=row['score']) for row in detail['filtered']]


def _aligned_footer(pixels, offset):
    if type(offset) is not int or offset not in (0, 13):
        raise PixelOcrError('unsupported original drink footer offset')
    aligned = np.zeros_like(pixels)
    aligned[1154:1244, :365] = pixels[1154+offset:1244+offset, :365]
    aligned.flags.writeable = False
    return aligned


def _native_detail(detail):
    raw = detail.raw_detail
    if isinstance(raw, str):
        raw = json.loads(raw)
    images = ([] if detail.raw_image is None else [detail.raw_image]) + list(detail.draw_images or [])
    return dict(reco_id=detail.reco_id, hit=detail.hit,
        box=list(detail.box) if detail.box is not None else None, raw_detail=raw,
        has_raw_image=detail.raw_image is not None, image_count=len(images)), images


def _child_python():
    # Match CPython's popen_spawn_win32 virtualenv handling, without spawn's
    # re-import of parent __main__. Track the actual process, not venv redirector.
    executable, env = sys.executable, None
    if os.name == 'nt' and executable != getattr(sys, '_base_executable', executable):
        env = os.environ.copy()
        env['__PYVENV_LAUNCHER__'] = executable
        executable = sys._base_executable
    return executable, env


class HifPixelOcrPool:
    """One in-flight batch, three fixed children, inherited deadline, no fallback.

    warm_up must finish before the caller takes its new three guarded captures.
    submit copies each source immediately; collect returns original submit order.
    Results retain the caller's timestamps, including old offline replay sources;
    this class never certifies their age or turns them into input permission.
    """

    def __init__(self, runtime_path, resource_paths, game_pid, clock, deadline_ms, *,
                 sdk_path, log_path, expected_resource_hash, expected_version='v5.12.3',
                 guard=lambda: None, slot_specs=None):
        self.clock, self.deadline_ms, self.guard = clock, deadline_ms, guard
        self.game_pid, self.pool_id = game_pid, uuid.uuid4().hex
        self.started_at_ms = clock.now()
        self._wall_deadline = time.monotonic() + max(0, deadline_ms - self.started_at_ms) / 1000
        self.closed, self.warmed, self.failure = False, False, None
        self._children, self._threads, self._logs, self.ready = [], [], [], []
        self._instance_ids = []
        self._inbox, self._requests, self._seen_frames = queue.Queue(), [], set()
        self._reco_ids, self._batch, self._batch_number = set(), None, 0
        self.partial_results = []
        self.binding = dict(version=VERSION, sdk_version=expected_version,
            runtime_path=str(Path(runtime_path).resolve()),
            resource_paths=[str(Path(p).resolve()) for p in resource_paths],
            sdk_path=str(Path(sdk_path).resolve()), resource_hash=expected_resource_hash,
            backend='CPU', ocr=OCR, slot_specs=json.loads(json.dumps(slot_specs)))
        try:
            if type(game_pid) is not int or game_pid <= 0 or not expected_resource_hash:
                raise PixelOcrError('invalid pixel pool identity')
            if slot_specs is not None:
                if len(slot_specs) != 4:
                    raise PixelOcrError('all four original drink nodes required')
                compact = [f'produce/hif_live_drink_{i}_480.png' for i in (1, 2, 3)]
                standard = [f'produce/drink_{i}.png' for i in (1, 2, 3)]
                for index, (row, x) in enumerate(zip(slot_specs, (17, 102, 187, 273), strict=True)):
                    params = row['parameters']
                    expected = dict(recognition='TemplateMatch', roi=[x, 1154, 90, 90],
                                    template=params['template'], green_mask=True, threshold=.8)
                    if (row != dict(name=f'LiveStatusDrinkSlot{index}', parameters=expected)
                            or params['template'] not in (compact, standard)):
                        raise PixelOcrError('drink recognition differs from original four nodes')
            self._check(deadline_ms)
            directory = Path(log_path).resolve() / self.pool_id
            directory.mkdir(parents=True, exist_ok=False)
            for index in range(WORKERS):
                log = (directory / f'worker-{index}.log').open('wb')
                self._logs.append(log)
                # A fixed file entry, NOT multiprocessing spawn. Neither normal
                # agent/main.py nor acceptance_agent.py is re-executed in children.
                executable, env = _child_python()
                child = subprocess.Popen([executable, '-u', str(Path(__file__).resolve()), '--pixel-worker'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), env=env)
                self._children.append(child)
                self._instance_ids.append(uuid.uuid4().hex)
                self._thread(self._read_child, index, child)
                self._send(index, dict(kind='init', pool_id=self.pool_id, worker=index,
                    instance_id=self._instance_ids[index], binding=self.binding,
                    log_path=str(directory / f'native-{index}')))
            self.ready = self._receive('ready', deadline_ms)
            for index, item in enumerate(self.ready):
                if (item['pool_id'] != self.pool_id or item['worker'] != index
                        or item['pid'] != self._children[index].pid
                        or item['binding'] != self.binding or item['controller_bound'] is not False
                        or item['instance_id'] != self._instance_ids[index]
                        or item['entry_file'] != str(Path(__file__).resolve())):
                    raise PixelOcrError('pixel worker initialization differs from fixed source')
            if len({row['instance_id'] for row in self.ready}) != WORKERS:
                raise PixelOcrError('pixel worker instances are not independent')
            self.ready_at_ms = clock.now()
        except BaseException as error:
            self.abort(str(error))
            raise

    def _thread(self, target, *args):
        thread = threading.Thread(target=target, args=args, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _read_child(self, index, child):
        try:
            while True:
                message, images = _read_packet(child.stdout)
                self._inbox.put((index, message, images))
        except (EOFError, OSError, ValueError, KeyError, PixelOcrError) as error:
            self._inbox.put((index, dict(kind='failure', reason=str(error)), []))

    def _send(self, index, message, images=()):
        def send():
            try:
                _write_packet(self._children[index].stdin, message, images)
            except (OSError, ValueError, PixelOcrError) as error:
                self._inbox.put((index, dict(kind='failure', reason=str(error)), []))
        self._thread(send)

    def _check(self, deadline_ms):
        if self.closed:
            raise PixelOcrError('pixel pool closed: ' + str(self.failure or 'scope ended'))
        self.guard()  # Local scope/epoch check supplied by the owning caller.
        remaining = min((min(deadline_ms, self.deadline_ms) - self.clock.now()) / 1000,
                        self._wall_deadline - time.monotonic())
        if remaining <= 0:
            raise PixelOcrError('pixel pool inherited deadline exhausted')
        return remaining

    def _receive(self, kind, deadline_ms):
        replies = {}
        while len(replies) < WORKERS:
            remaining = self._check(deadline_ms)
            try:
                index, item, images = self._inbox.get(timeout=min(remaining, .05))
            except queue.Empty:
                if any(child.poll() is not None for child in self._children):
                    raise PixelOcrError('pixel worker exited during batch')
                continue
            if index in replies or item.get('kind') != kind:
                raise PixelOcrError('pixel worker partial/duplicate/failure: ' + str(item))
            if kind == 'result':
                self._validate(index, item)
                details = [item] + item['slot_results']
                if len(images) != sum(row['image_count'] for row in details):
                    raise PixelOcrError('incomplete pixel recognition images')
                offset = 0
                for row in details:
                    end = offset + row['image_count']
                    row['raw_image'] = images[offset] if row['has_raw_image'] else None
                    row['draw_images'] = images[offset+int(row['has_raw_image']):end]
                    offset = end
                self.partial_results.append(item)
            elif images:
                raise PixelOcrError('unexpected initialization pixels')
            replies[index] = item
        self._check(deadline_ms)
        if any(child.poll() is not None for child in self._children):
            raise PixelOcrError('pixel worker exited before batch acceptance')
        return [replies[index] for index in range(WORKERS)]

    def _validate(self, index, item):
        request, worker = self._requests[index], self.ready[index]
        for key in ('pool_id', 'batch_id', 'phase', 'request_id', 'frame_id', 'source_frame_id',
                    'captured_at_ms', 'game_pid', 'pixel_sha256', 'footer_offset', 'aligned_pixel_sha256'):
            if item.get(key) != request[key]:
                raise PixelOcrError('pixel result source mismatch: ' + key)
        if (item.get('worker') != index or item.get('pid') != worker['pid']
                or item.get('instance_id') != worker['instance_id'] or item.get('binding') != self.binding):
            raise PixelOcrError('pixel result worker/model changed')
        reco_id = item.get('reco_id')
        scoped_id = (worker['instance_id'], reco_id)
        if type(reco_id) is not int or reco_id <= 0 or scoped_id in self._reco_ids:
            raise PixelOcrError('pixel result native identity missing or reused')
        if not isinstance(item.get('raw_detail'), dict) or item.get('rows') != _rows(item['raw_detail']):
            raise PixelOcrError('pixel result rows differ from complete native detail')
        if type(item.get('hit')) is not bool:
            raise PixelOcrError('pixel result lacks native hit status')
        self._reco_ids.add(scoped_id)
        slots = item.get('slot_results')
        expected = self.binding['slot_specs'] if request['footer_offset'] is not None else []
        if not isinstance(slots, list) or len(slots) != len(expected):
            raise PixelOcrError('pixel result lacks complete original drink templates')
        for result, spec in zip(slots, expected, strict=True):
            key = (worker['instance_id'], result.get('reco_id'))
            if (any(result.get(name) != value for name, value in spec.items())
                    or type(result.get('reco_id')) is not int or result['reco_id'] <= 0 or key in self._reco_ids
                    or type(result.get('hit')) is not bool or not isinstance(result.get('raw_detail'), dict)
                    or not isinstance(result['raw_detail'].get('filtered'), list)):
                raise PixelOcrError('pixel drink result identity/parameters/detail changed')
            self._reco_ids.add(key)

    def _submit(self, frame_id, captured_at_ms, image, deadline_ms, phase, footer_offset=None):
        self._check(deadline_ms)
        if (len(self._requests) >= WORKERS or not isinstance(frame_id, str) or not frame_id
                or frame_id in self._seen_frames or type(captured_at_ms) is not int
                or captured_at_ms < 0 or captured_at_ms > self.clock.now()):
            raise PixelOcrError('invalid/duplicate pixel capture source')
        if self._requests and captured_at_ms < self._requests[-1]['captured_at_ms']:
            raise PixelOcrError('pixel captures are out of source order')
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (1280, 720, 3):
            raise PixelOcrError('full 720x1280 BGR pixels required')
        pixels = np.array(image, copy=True, order='C')
        pixels.flags.writeable = False
        aligned_hash = None
        if footer_offset is not None:
            if not self.binding['slot_specs']:
                raise PixelOcrError('pixel pool has no original drink node configuration')
            aligned_hash = hashlib.sha256(_aligned_footer(pixels, footer_offset).tobytes()).hexdigest()
        if not self._requests:
            self._batch_number += 1
            self._batch = f'{self.pool_id}:{self._batch_number}'
            self.partial_results = []
        index = len(self._requests)
        message = dict(kind='recognize', phase=phase, pool_id=self.pool_id,
            batch_id=self._batch, request_id=f'{self._batch}:{index}', worker=index,
            frame_id=frame_id, source_frame_id=frame_id, captured_at_ms=captured_at_ms, game_pid=self.game_pid,
            footer_offset=footer_offset, aligned_pixel_sha256=aligned_hash,
            pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
        self._requests.append(message)
        self._seen_frames.add(frame_id)
        self._send(index, message, [pixels])

    def warm_up(self, preparation_pixels, deadline_ms, *, footer_offset=None):
        try:
            if self.warmed or self._requests:
                raise PixelOcrError('pixel warm-up may run only once before captures')
            now = self.clock.now()
            for index in range(WORKERS):
                self._submit(f'warm-only:{self.pool_id}:{index}', now, preparation_pixels, deadline_ms, 'warm', footer_offset)
            self._receive('result', deadline_ms)
            self._requests, self.partial_results = [], []
            self.warmed, self.warmed_at_ms = True, self.clock.now()
        except BaseException as error:
            self.abort(str(error))
            raise

    def submit(self, frame_id, captured_at_ms, image, deadline_ms, *, footer_offset=None):
        try:
            if not self.warmed:
                raise PixelOcrError('pixel pool must warm before fresh captures')
            self._submit(frame_id, captured_at_ms, image, deadline_ms, 'observe', footer_offset)
        except BaseException as error:
            self.abort(str(error))
            raise

    def collect(self, deadline_ms):
        try:
            if len(self._requests) != WORKERS:
                raise PixelOcrError('all three original captures are required')
            result = self._receive('result', deadline_ms)
            self._requests = []
            return result
        except BaseException as error:
            self.abort(str(error))
            raise

    def abort(self, reason):
        self.failure = reason
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        pending = bool(self._requests)
        for child in self._children:
            if child.poll() is None:
                if pending:
                    child.terminate()
                else:
                    try:
                        _write_packet(child.stdin, dict(kind='stop'))
                    except (OSError, ValueError):
                        child.terminate()
        until = time.monotonic() + 1
        for child in self._children:
            try:
                child.wait(timeout=max(0, until - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=1)
            child.stdin.close()
            child.stdout.close()
        for log in self._logs:
            log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _worker():
    # Native/library stdout goes to the private log; only the duplicated pipe
    # carries framed data. No parent __main__ or registration module is imported.
    output = os.fdopen(os.dup(sys.stdout.fileno()), 'wb')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    incoming = sys.stdin.buffer
    tasker = resource = dll = None
    try:
        init, images = _read_packet(incoming)
        if init['kind'] != 'init' or images:
            raise PixelOcrError('pixel worker needs fixed initialization')
        binding = init['binding']
        sys.path.insert(0, binding['sdk_path'])
        from maa.tasker import Tasker
        from maa.library import Library
        from maa.toolkit import Toolkit
        from maa.pipeline import JOCR, JTemplateMatch
        from maa.resource import Resource

        dll = os.add_dll_directory(binding['runtime_path'])
        Library.open(Path(binding['runtime_path']))
        if Library.version() != binding['sdk_version'] or binding['ocr'] != OCR or binding['backend'] != 'CPU':
            raise PixelOcrError('pixel worker SDK/recognition configuration differs')
        if not Toolkit.init_option(init['log_path'], dict(logging=True, save_draw=False, save_on_error=False)):
            raise PixelOcrError('pixel worker logging initialization failed')
        resource = Resource()
        if not resource.use_cpu():
            raise PixelOcrError('pixel worker CPU backend rejected')
        for path in binding['resource_paths']:
            if not resource.post_bundle(Path(path)).wait().succeeded:
                raise PixelOcrError('pixel worker resource load failed')
        if resource.hash != binding['resource_hash']:
            raise PixelOcrError('pixel worker resource revision differs')
        tasker = Tasker()
        if (not Library.framework().MaaTaskerBindResource(tasker._handle, resource._handle)
                or Library.framework().MaaTaskerGetController(tasker._handle) or not tasker.inited):
            raise PixelOcrError('pixel worker must be resource-only')
        identity = dict(pool_id=init['pool_id'], worker=init['worker'], pid=os.getpid(),
            instance_id=init['instance_id'], binding=binding, controller_bound=False,
            entry_file=str(Path(__file__).resolve()))
        _write_packet(output, dict(kind='ready', **identity))
        while True:
            message, images = _read_packet(incoming)
            if message['kind'] == 'stop':
                break
            if (message['kind'] != 'recognize' or message['pool_id'] != init['pool_id']
                    or message['worker'] != init['worker'] or len(images) != 1):
                raise PixelOcrError('pixel worker unexpected request')
            pixels = images[0]
            if (pixels.shape != (1280, 720, 3)
                    or hashlib.sha256(pixels.tobytes()).hexdigest() != message['pixel_sha256']):
                raise PixelOcrError('pixel worker input bytes differ')
            began = time.perf_counter_ns()
            job = tasker.post_recognition('OCR', JOCR(**OCR), pixels)
            job.wait()
            waited = time.perf_counter_ns()
            if not job.succeeded:
                raise PixelOcrError('pixel worker recognition failed')
            result = job.get()
            obtained = time.perf_counter_ns()
            if len(result.nodes) != 1 or result.nodes[0].recognition is None:
                raise PixelOcrError('pixel worker recognition incomplete')
            detail = result.nodes[0].recognition
            if hashlib.sha256(pixels.tobytes()).hexdigest() != message['pixel_sha256']:
                raise PixelOcrError('pixel worker source mutated')
            full, native_images = _native_detail(detail)
            slots = []
            if message['footer_offset'] is not None:
                aligned = _aligned_footer(pixels, message['footer_offset'])
                if hashlib.sha256(aligned.tobytes()).hexdigest() != message['aligned_pixel_sha256']:
                    raise PixelOcrError('pixel worker aligned footer source differs')
                if not binding['slot_specs']:
                    raise PixelOcrError('pixel worker lacks original slot configuration')
                for spec in binding['slot_specs']:
                    started = time.perf_counter_ns()
                    params = dict(spec['parameters'])
                    del params['recognition']
                    params['threshold'] = [params['threshold']]
                    job = tasker.post_recognition('TemplateMatch', JTemplateMatch(**params), aligned)
                    job.wait()
                    if not job.succeeded:
                        raise PixelOcrError('pixel worker original drink template failed')
                    result = job.get()
                    if len(result.nodes) != 1 or result.nodes[0].recognition is None:
                        raise PixelOcrError('pixel worker drink template incomplete')
                    value, saved_images = _native_detail(result.nodes[0].recognition)
                    value.update(spec, elapsed_ms=(time.perf_counter_ns()-started)/1e6)
                    slots.append(value)
                    native_images.extend(saved_images)
            reply = dict(message, **{key: value for key, value in identity.items() if key not in message})
            reply.update(full, kind='result', rows=_rows(full['raw_detail']), slot_results=slots,
                         wait_ms=(waited-began)/1e6, get_ms=(obtained-waited)/1e6)
            _write_packet(output, reply, native_images)
    except BaseException as error:
        _write_packet(output, dict(kind='failure', reason=f'{type(error).__name__}: {error}'))
    finally:
        if tasker is not None:
            del tasker
        if resource is not None:
            del resource
        if dll is not None:
            dll.close()
        output.close()


if __name__ == '__main__' and sys.argv[1:] == ['--pixel-worker']:
    _worker()
