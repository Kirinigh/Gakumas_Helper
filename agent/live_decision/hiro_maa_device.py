"""The active Maa task as a Hiro device."""

import json
import time
import sqlite3
from types import SimpleNamespace
from pathlib import Path
from contextlib import contextmanager
from importlib.util import find_spec

from .hiro_runtime import hiro_module, hiro_binding, hiro_route_file


def _mounted_master():
    from .high_score.tower_gnrpa import load_native_tower

    _, upstream = load_native_tower()
    master = upstream.master()
    db = sqlite3.connect(':memory:')
    tables = ('Character', 'IdolCard', 'MemoryAbility', 'ProduceCard', 'ProduceCardCustomize',
              'ProduceDrink', 'ProduceExamEffect', 'ProduceItem', 'SupportCard')
    for table in tables:
        rows = master.rows(table)
        if not rows:
            continue
        columns = list(dict.fromkeys(key for row in rows for key in row))
        names = ', '.join(f'"{column}"' for column in columns)
        placeholders = ', '.join('?' for _ in columns)
        db.execute(f'CREATE TABLE "{table}" ({names})')
        db.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})', (
            tuple(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                  for value in (row.get(key) for key in columns)) for row in rows))
    return db


@contextmanager
def _master_source():
    """Use the same pinned master as Maa's exam host when KAA DB is absent."""
    assets = hiro_module('vision.assets').source()
    original = getattr(assets, 'master_db', None)
    if original is None or original() is not None:
        yield
        return
    catalog = hiro_module('state.catalog')
    cached = getattr(catalog._LOCAL, 'catalog', None)
    db = None

    def master_db():
        nonlocal db
        if db is None:
            db = _mounted_master()
        return db

    assets.master_db = master_db
    catalog._LOCAL.catalog = None
    clear = hiro_module('master').clear
    clear()
    try:
        yield
    finally:
        assets.master_db = original
        catalog._LOCAL.catalog = cached
        clear()
        if db is not None:
            db.close()


class HiroMaaDevice:
    name = 'maa'

    def __init__(self, context, authorise, *, log):
        self.context = context
        self.authorise = authorise
        self.log = log
        self.screen = hiro_module('devices.base').LogicalScreen()
        self.sequence = 0

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
            raise hiro_module('devices.base').DeviceUnavailable('Maa screen capture failed')
        return job.get()

    def screenshot(self):
        image = hiro_module('devices.base').settled_frame(self._capture, self.screen)
        self.sequence += 1
        return image

    def _dispatch(self, name, point, *args):
        self._guard()
        controller = self.context.tasker.controller
        job, result = None, None
        try:
            job = getattr(controller, name)(*args)
            result = job.wait()
        except (OSError, RuntimeError) as error:
            raise hiro_module('devices.base').DeviceError('Maa input outcome is unknown') from error
        finally:
            success = getattr(result, 'succeeded', None) is True
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
def hiro_maa_route(device, read_ocr, *, session=None, clock=time.monotonic, sleep=time.sleep):
    """Run the original page handlers with the current Maa OCR and UI templates."""
    pack = hiro_route_file(Path('assets/pack/ui/templates.json')).parent
    ui = hiro_module('vision.ui_templates')
    ocr = hiro_module('vision.ocr')
    pages = hiro_module('vision.classify')
    digits = hiro_module('vision.digits')
    rt = hiro_module('runtime.loop').Runtime(raw_device=device, classify=pages.classify,
                                            session=session, clock=clock, sleep=sleep)

    def read_bgr(img, scale, languages):
        return [(row.text, row.score,
                 tuple(round(v / scale) for v in (x, y, x + width, y + height)))
                for row in read_ocr(img) for x, y, width, height in [row.box]]

    with hiro_binding(rt), _master_source():
        old = ui.PACK, pages.PACK, ocr.engine, ocr._rapid_raw, digits._dddd
        caches = (ui._catalog, ui._pixels, ui.tpl)
        try:
            ui.PACK = pages.PACK = pack
            ocr.engine = lambda: 'rapidocr'
            ocr._rapid_raw = read_bgr
            if find_spec('ddddocr') is None:
                # The installed Maa agent supplies OCR. Mark the optional
                # reader as absent so the original vote cannot count it.
                digits._dddd = lambda: SimpleNamespace(classification=lambda _: '')
            for cache in caches:
                cache.cache_clear()
            yield rt
        finally:
            ui.PACK, pages.PACK, ocr.engine, ocr._rapid_raw, digits._dddd = old
            for cache in caches:
                cache.cache_clear()
