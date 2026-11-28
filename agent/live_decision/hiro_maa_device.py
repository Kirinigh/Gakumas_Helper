"""The active Maa task as a Hiro device."""

import time
from types import SimpleNamespace
from pathlib import Path
from contextlib import contextmanager
from importlib.util import find_spec

import cv2

from .simd_ocr import read_optional_simd_ocr
from .hiro_runtime import hiro_module, hiro_binding, hiro_route_file
from .hif_capture_adapter import MAA_CAPTURE_ERROR_PREFIX


class HiroMaaDevice:
    name = 'maa'

    def __init__(self, context, authorise, *, log):
        self.context = context
        self.authorise = authorise
        self.log = log
        self.screen = hiro_module('devices.base').LogicalScreen()
        self.sequence = 0
        self._uncertain_input = False

    def connect(self):
        self._guard()

    def close(self):
        # The Maa task owns the controller across both HIF segments.
        pass

    @property
    def orientation(self):
        return self.screen.orientation

    def alive(self):
        return not self.context.tasker.stopping

    def _guard(self):
        unavailable = hiro_module('devices.base').DeviceUnavailable
        if not self.alive():
            raise unavailable('The Maa task has stopped')
        try:
            self.authorise()
        except ValueError as error:
            raise unavailable(str(error)) from error

    def _capture(self):
        self._guard()
        job = self.context.tasker.controller.post_screencap().wait()
        if job.succeeded is not True:
            raise hiro_module('devices.base').DeviceUnavailable('控制器未成功截图')
        return job.get()

    def screenshot(self):
        geometry = hiro_module('devices.base')
        try:
            image = geometry.settled_frame(self._capture, self.screen)
        except geometry.DeviceUnavailable as error:
            raise geometry.DeviceUnavailable(f'{MAA_CAPTURE_ERROR_PREFIX}{error}') from error
        self.sequence += 1
        return image

    def _dispatch(self, name, point, *args):
        self._guard()
        if self._uncertain_input:
            raise hiro_module('devices.base').DeviceError('Maa input outcome is unknown')
        controller = self.context.tasker.controller
        job, result = None, None
        try:
            job = getattr(controller, name)(*args)
            result = job.wait()
        except (OSError, RuntimeError) as error:
            raise hiro_module('devices.base').DeviceError('Maa input outcome is unknown') from error
        finally:
            success = getattr(result, 'succeeded', None) is True
            # Some upstream helpers catch device errors and try a fallback.
            # Once dispatch is uncertain, this task's read access can remain
            # available without sending a second action through the controller.
            self._uncertain_input = not success
            self.log(dict(event='hiro_maa_input', frame=self.sequence, operation=name, point=point,
                          job_id=getattr(job, 'job_id', None), sdk_success=True if success else None))
        if not success:
            # The action may have reached the client. Exit the upstream
            # runtime rather than let its recognition loop submit it again.
            raise hiro_module('devices.base').DeviceError('Maa input outcome is unknown')
        return job.job_id

    def tap(self, x, y):
        return self._dispatch('post_click', (x, y), *self.screen.to_native(x, y))

    def swipe(self, x1, y1, x2, y2, duration=.3, *, hold=0.):
        start = self.screen.to_native(x1, y1)
        end = self.screen.to_native(x2, y2)
        if hold:
            # Maa has no extra hold argument for a swipe. Hold time is part of
            # the gesture duration on the controller, which owns the input.
            duration += hold
        return self._dispatch('post_swipe', (x1, y1, x2, y2), *start, *end, round(duration * 1000))


@contextmanager
def hiro_maa_ocr(read_ocr):
    """Bind the upstream box and single-line OCR paths inside a Hiro device scope."""
    ocr = hiro_module('vision.ocr')
    simd = hiro_module('vision.simd_ocr')
    digits = hiro_module('vision.digits')

    def read_bgr(img, *, scale, only_rec=False):
        height, width = img.shape[:2]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        scale = min(scale, 2000 / max(height, width))
        if scale != 1:
            img = cv2.resize(img, (max(1, round(width * scale)), max(1, round(height * scale))),
                             interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        rows = read_optional_simd_ocr(img, {'only_rec': True}) if only_rec else ()
        sx, sy = width / img.shape[1], height / img.shape[0]
        return [(row.text, row.score, (round(x * sx), round(y * sy),
                                      round((x + w) * sx), round((y + h) * sy)))
                for row in rows or read_ocr(img) for x, y, w, h in [row.box]]

    old = ocr.engine, simd.simd_engine, digits._dddd
    try:
        ocr.engine = lambda: 'simd'
        worker = SimpleNamespace(read=read_bgr)
        simd.simd_engine = lambda: worker
        if find_spec('ddddocr') is None:
            # The installed Maa agent supplies OCR. Mark the optional
            # reader as absent so the original vote cannot count it.
            digits._dddd = lambda: SimpleNamespace(classification=lambda _: '')
        yield
    finally:
        ocr.engine, simd.simd_engine, digits._dddd = old


@contextmanager
def hiro_maa_route(device, read_ocr, *, session=None, clock=time.monotonic, sleep=time.sleep):
    """Run the original page handlers with the current Maa OCR and UI templates."""
    pack = hiro_route_file(Path('assets/pack/ui/templates.json')).parent
    ui = hiro_module('vision.ui_templates')
    pages = hiro_module('vision.classify')
    rescue = hiro_module('loop.rescue')
    rt = hiro_module('runtime.loop').Runtime(raw_device=device, classify=pages.classify,
                                            session=session, clock=clock, sleep=sleep, rescue=False)

    with hiro_binding(rt), hiro_maa_ocr(read_ocr):
        old = ui.PACK, pages.PACK, rescue.LLM_UNKNOWN
        caches = (ui._catalog, ui._pixels, ui.tpl)
        try:
            ui.PACK = pages.PACK = pack
            rescue.LLM_UNKNOWN = False
            for cache in caches:
                cache.cache_clear()
            yield rt
        finally:
            ui.PACK, pages.PACK, rescue.LLM_UNKNOWN = old
            for cache in caches:
                cache.cache_clear()
