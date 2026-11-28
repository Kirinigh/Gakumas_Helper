"""Shared SimdPaddleOCR worker for live pixel-only recognition."""

import os
import re
import json
import atexit
import base64
import shutil
import logging
import subprocess
from time import monotonic
from queue import Empty, Queue
from types import SimpleNamespace
from pathlib import Path
from functools import cache, partial
from threading import Lock, Timer, Thread
from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import Future

import cv2
import numpy as np

from agent.utils.runtime_paths import resolve_packaged_runtime_path as resolve_runtime_path

VULKAN_COLD_WAIT_SECONDS = 1.0
_wait_for_vulkan_cold = ContextVar("wait_for_vulkan_cold", default=True)
_vulkan_setup_lock = Lock()


@contextmanager
def without_vulkan_cold_wait():
    """Keep the first Vulkan exchange off a caller's observation deadline."""
    token = _wait_for_vulkan_cold.set(False)
    try:
        yield
    finally:
        _wait_for_vulkan_cold.reset(token)


class OcrWorker:
    _start_timeout = 10.0
    _reply_timeout = 3.0
    _disable_on_transport_failure = True

    def __init__(self, command):
        self.command = command
        self.process = None
        self.lock = Lock()
        self.available = True
        atexit.register(self.close)

    @staticmethod
    def _receive(stream, replies):
        try:
            for line in stream:
                replies.put(line)
        finally:
            replies.put(None)

    def _read(self, timeout):
        try:
            reply = self.replies.get(timeout=timeout)
        except Empty as error:
            raise RuntimeError('OCR worker did not reply') from error
        if reply is None:
            raise RuntimeError('OCR worker exited')
        try:
            result = json.loads(reply)
        except ValueError as error:
            raise RuntimeError('OCR worker returned an invalid reply') from error
        if not isinstance(result, dict):
            raise RuntimeError('OCR worker returned an invalid reply')
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def _start(self):
        self.replies = Queue()
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if not self.available:
            raise RuntimeError('OCR worker is unavailable')
        Thread(target=self._receive, args=(self.process.stdout, self.replies), daemon=True).start()
        if self._read(self._start_timeout).get('ready') is not True:
            raise RuntimeError('OCR worker is not ready')

    def _close_process(self):
        process = self.process
        if process is None:
            return
        self.process = None
        try:
            if process.poll() is None:
                process.kill()
            process.wait()
        finally:
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass

    def close(self):
        with self.lock:
            self._close_process()

    def request_pixels(self, pixels, **options):
        height, width = pixels.shape[:2]
        request = dict(width=width, height=height, **options,
                       pixels=base64.b64encode(np.ascontiguousarray(pixels)).decode('ascii'))
        with self.lock:
            if not self.available:
                raise RuntimeError('OCR worker is unavailable')
            try:
                if self.process is None:
                    self._start()
                    if not self.available:
                        raise RuntimeError('OCR worker is unavailable')
                self.process.stdin.write(json.dumps(request) + '\n')
                self.process.stdin.flush()
                return self._read(self._reply_timeout)
            except (OSError, RuntimeError, KeyError):
                if self._disable_on_transport_failure:
                    self.available = False
                self._close_process()
                raise


class SimdOcr(OcrWorker):
    def recognize(self, pixels, *, only_rec):
        return self.request_pixels(pixels, only_rec=only_rec)['lines']

    def read(self, image, *, roi=None, expected='.+', only_rec=False, threshold=0.3,
             replace=(), order_by='Vertical'):
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
                or image.dtype != np.uint8):
            raise ValueError('expected a BGR image')
        x, y, width, height = roi or (0, 0, image.shape[1], image.shape[0])
        frame = image[y:y + height, x:x + width]
        if frame.shape[:2] != (height, width) or not frame.size:
            raise ValueError('invalid OCR crop')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if roi is not None and 2 * max(width, height) <= 2048:
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        pixels = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        patterns = (expected,) if isinstance(expected, str) else expected
        rows = []
        for line in self.recognize(pixels, only_rec=only_rec):
            value = line['Text'].strip()
            score = float(line['Score'])
            for source, target in replace:
                value = re.sub(source, target, value)
            if not value or not threshold <= score <= 1 or (patterns and not any(
                    re.search(pattern, value) for pattern in patterns)):
                continue
            left, right = (min(width, max(0, round(line[key] * width / pixels.shape[1])))
                           for key in ('Left', 'Right'))
            top, bottom = (min(height, max(0, round(line[key] * height / pixels.shape[0])))
                           for key in ('Top', 'Bottom'))
            if right > left and bottom > top:
                rows.append(SimpleNamespace(text=value, score=score,
                                            box=(x + left, y + top, right - left, bottom - top)))
        axes = (0, 1) if order_by == 'Horizontal' else (1, 0)
        return tuple(sorted(rows, key=lambda row: (row.box[axes[0]], row.box[axes[1]])))


class VulkanOcr(SimdOcr):
    """Bound the whole GPU exchange, including OS calls and pipe writes."""

    _start_timeout = 45.0
    _reply_timeout = 15.0
    _disable_on_transport_failure = False  # The exchange owns retry and availability.

    def __init__(self, command):
        super().__init__(command)
        self._dispatch_lock = Lock()
        self._pending = None
        self._warm = False
        self._exchange_deadline = None

    def _read(self, timeout):
        deadline = self._exchange_deadline
        remaining = max(0.0, deadline - monotonic()) if deadline is not None else timeout
        return super()._read(min(timeout, remaining))

    def _kill_process(self):
        process = self.process
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
            except OSError:
                pass

    def close(self):
        self.available = False
        self._kill_process()
        if self.lock.acquire(blocking=False):
            try:
                self._close_process()
            finally:
                self.lock.release()

    def _expire(self, pending):
        with self._dispatch_lock:
            if self._pending is not pending:
                return
            self.available = False
            self._pending = None
            pending.set_exception(RuntimeError('Vulkan OCR exchange timed out'))
        # Cleanup can itself wait on the OS. The CPU fallback must be free to run.
        Thread(target=self.close, daemon=True).start()

    def _exchange(self, pending, timer, deadline, operation):
        rows, error = (), None
        self._exchange_deadline = deadline
        try:
            rows = operation()
        except Exception as caught:
            error = caught
        finally:
            self._exchange_deadline = None
        timer.cancel()
        if monotonic() >= deadline:
            self._expire(pending)
            return

        fatal = isinstance(error, (OSError, RuntimeError, KeyError))
        with self._dispatch_lock:
            if self._pending is not pending:
                return
            self._pending = None
            if error is None:
                self._warm = True
                pending.set_result(rows)
            else:
                if fatal:
                    self.available = False
                pending.set_exception(error)
        if fatal:
            self.close()

    def _submit(self, timeout, operation):
        deadline = monotonic() + timeout
        pending = Future()
        self._pending = pending
        timer = Timer(timeout, self._expire, args=(pending,))
        timer.daemon = True
        timer.start()
        Thread(target=self._exchange, args=(pending, timer, deadline, operation), daemon=True).start()
        return pending, deadline

    def _read_cold(self, image, **params):
        try:
            return super().read(image, **params)
        except (OSError, RuntimeError):
            if not self.available or monotonic() >= self._exchange_deadline:
                raise
            # The failed transport has been closed. Retry the same frame within the original deadline.
            return super().read(image, **params)

    def read(self, image, **params):
        """Return None when this frame has no GPU reply, or the completed rows."""
        with self._dispatch_lock:
            if not self.available or self._pending is not None:
                return None
            cold = not self._warm
            operation = self._read_cold if cold else super().read
            timeout = self._start_timeout + self._reply_timeout if cold else 3.0
            wait_until = monotonic() + VULKAN_COLD_WAIT_SECONDS if cold else None
            pending, deadline = self._submit(timeout, partial(operation, image, **params))
        if cold and not _wait_for_vulkan_cold.get():
            return None
        wait_until = min(wait_until, deadline) if wait_until is not None else deadline
        try:
            return pending.result(timeout=max(0.0, wait_until - monotonic()))
        except TimeoutError:
            if wait_until < deadline and monotonic() < deadline and not pending.done():
                return None
            self._expire(pending)
            return pending.result()


def configured_simd_ocr(backend='cpu'):
    if backend != 'vulkan':
        return _configured_simd_ocr(backend)
    # The cache alone permits duplicate first calls from different threads.
    with _vulkan_setup_lock:
        return _configured_simd_ocr(backend)


@cache
def _configured_simd_ocr(backend):
    model = os.getenv('GKH_SIMD_OCR_MODEL', 'small')
    if model not in ('small', 'medium'):
        return None
    helper = resolve_runtime_path(Path(f'.local/task275-ocr/worker-{model}/GkhSimdOcr.dll'),
                                  Path('data/ocr/GkhSimdOcr.dll'))
    dotnet = shutil.which('dotnet')
    if helper.is_file() and dotnet:
        worker_type = VulkanOcr if backend == 'vulkan' else SimdOcr
        return worker_type((dotnet, str(helper.resolve()), backend, model))
    return None


def read_optional_simd_ocr(image, params, *, prefer_gpu=False, index=0, accept=bool):
    if (params.keys() - {'roi', 'expected', 'only_rec', 'threshold', 'replace', 'order_by'}
            or params.get('order_by', 'Vertical') not in ('Horizontal', 'Vertical')):
        return ()
    mode = os.getenv('GKH_SIMD_OCR', 'auto')
    if mode not in ('auto', 'cpu', 'vulkan'):
        return ()
    backends = ('vulkan', 'cpu') if mode == 'vulkan' or (mode == 'auto' and prefer_gpu) else ('cpu',)
    for backend in backends:
        worker = configured_simd_ocr(backend)
        if worker is None or not worker.available:
            continue
        try:
            rows = worker.read(image, **params)
        except (TypeError, ValueError, re.error):
            return ()
        except (OSError, RuntimeError, KeyError) as error:
            worker.available = False
            logging.getLogger(__name__).warning('SIMD OCR %s unavailable: %s', backend, error)
            continue
        if rows is None:
            continue
        if -len(rows) <= index < len(rows) and accept(rows):
            return rows
        if backend == 'vulkan':
            # The GPU completed this frame. Use the caller's native fallback
            # when its rows are empty or fail the caller's evidence checks.
            break
    return ()
