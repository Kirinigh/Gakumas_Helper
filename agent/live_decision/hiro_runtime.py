"""Load the pinned Hiro implementation used by the Maa policy route."""

import os
import sys
from pathlib import Path
from functools import cache
from importlib import import_module
from threading import RLock
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location

from agent.utils.runtime_paths import resolve_packaged_runtime_path as resolve_runtime_path

HIRO_REVISION = '6e7bbd59a7ca2a66c1f1615cb9bd8e2e3b3ede93'
HIRO_MASTER_REVISION = 'edc305aac3ac0204806c96164ce36fb795ba81e1'
_RUNTIME_OWNER = RLock()


def hiro_runtime_root():
    return resolve_runtime_path(Path('.local/task275-rust/runtime') / HIRO_REVISION,
                                Path('data/idol_road/rust_gnrpa') / HIRO_REVISION)


@cache
def hiro_source():
    runtime = hiro_runtime_root()
    source = runtime / 'src'
    exam = source / 'hiro/exec/exam.py'
    if not exam.is_file():
        raise ImportError(f'Hiro {HIRO_REVISION} exam source unavailable: {exam}')
    os.environ['GAKUMAS_PLAYER'] = str(runtime / 'vendor/gakumas-player')
    os.environ['GAKUMAS_MASTER_COMMIT'] = HIRO_MASTER_REVISION
    os.environ['HIRO_TOWER_BINDINGS'] = str(runtime / 'tower_bindings.json')
    os.environ['GAKUMAS_EXAM_RS_CACHE'] = str(runtime / 'native-master')
    if not os.environ.get('HIRO_ASSET_SOURCE'):
        os.environ['HIRO_ASSET_SOURCE'] = 'agent.live_decision.hiro_master_assets:mounted_asset_source'
    os.environ['HIRO_KAA_GAME_DATA'] = str(runtime / 'resources/game_data')
    os.environ['HIRO_BUNDLED_ASSETS'] = str(runtime / 'assets/pack')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    package = import_module('hiro')
    # The Rust-only subset can already be imported by a strategy provider.
    python_package = str(source / 'hiro')
    if python_package not in package.__path__:
        package.__path__.insert(0, python_package)
    return source


def hiro_module(name):
    source = hiro_source() / 'hiro'
    parent = 'hiro'
    for part in name.split('.')[:-1]:
        parent += '.' + part
        source /= part
        package = import_module(parent)
        # The native strategy ships a smaller bench_bridge package and may
        # import it before the page runner needs the rest of that package.
        if source.is_dir() and str(source) not in package.__path__:
            package.__path__ = [str(source), *package.__path__]
    return import_module(f'hiro.{name}')


def hiro_route_file(relative):
    reference = hiro_runtime_root() / relative
    if not reference.is_file():
        raise FileNotFoundError(f'Hiro {HIRO_REVISION} route file is unavailable: {relative}')
    return reference


@contextmanager
def hiro_ui_templates():
    """Use the same pinned template pack in and outside a Hiro page runner."""
    with _RUNTIME_OWNER:
        ui = hiro_module('vision.ui_templates')
        pack = hiro_route_file(Path('assets/pack/ui/templates.json')).parent
        previous = ui.PACK
        if pack == previous:
            yield ui
            return
        caches = (ui._catalog, ui._pixels, ui.tpl)
        ui.PACK = pack
        try:
            for reader in caches:
                reader.cache_clear()
            yield ui
        finally:
            ui.PACK = previous
            for reader in caches:
                reader.cache_clear()


@cache
def hiro_hif_campaign():
    script = hiro_route_file(Path('tools/play_game.py'))
    spec = spec_from_file_location('_gkh_pinned_hiro_play_game', script)
    module = module_from_spec(spec)
    directory = str(script.parent)
    occurrences = sys.path.count(directory)
    try:
        spec.loader.exec_module(module)
    finally:
        if sys.path.count(directory) > occurrences:
            sys.path.remove(directory)
    return module


@contextmanager
def hiro_binding(binding):
    """Bind Hiro's active device for a route or for an exam inside that route."""
    with _RUNTIME_OWNER:
        active = hiro_module('devices.active')
        parent = active._RT
        active.use(binding)
        try:
            yield
        finally:
            active.use(parent)
