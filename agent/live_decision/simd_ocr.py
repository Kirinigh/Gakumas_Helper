"""Shared SimdPaddleOCR worker for live pixel-only recognition."""

import os
import re
import json
import atexit
import base64
import shutil
import logging
import subprocess
from queue import Empty, Queue
from types import SimpleNamespace
from pathlib import Path
from functools import cache
from threading import Lock, Thread

import cv2
import numpy as np

from agent.utils.runtime_paths import resolve_packaged_runtime_path as resolve_runtime_path


class OcrWorker:
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
        Thread(target=self._receive, args=(self.process.stdout, self.replies), daemon=True).start()
        if self._read(10).get('ready') is not True:
            raise RuntimeError('OCR worker is not ready')

    def _close_process(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None

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
                self.process.stdin.write(json.dumps(request) + '\n')
                self.process.stdin.flush()
                return self._read(3)
            except (OSError, RuntimeError, KeyError):
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


@cache
def configured_simd_ocr(backend='cpu'):
    model = os.getenv('GKH_SIMD_OCR_MODEL', 'small')
    if model not in ('small', 'medium'):
        return None
    helper = resolve_runtime_path(Path(f'.local/task275-ocr/worker-{model}/GkhSimdOcr.dll'),
                                  Path('data/ocr/GkhSimdOcr.dll'))
    dotnet = shutil.which('dotnet')
    if helper.is_file() and dotnet:
        return SimdOcr((dotnet, str(helper.resolve()), backend, model))
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
        if -len(rows) <= index < len(rows) and accept(rows):
            return rows
    return ()
