"""Load the pinned Hiro implementation used by the Maa policy route."""

import sys
from pathlib import Path
from functools import cache
from importlib import import_module
from threading import RLock
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location

from agent.utils.runtime_paths import resolve_packaged_runtime_path as resolve_runtime_path

HIRO_REVISION = '8380622a467cc30296c91b79dba4348166ef8d12'
_RUNTIME_OWNER = RLock()


@cache
def hiro_source():
    installed = Path('data/idol_road/rust_gnrpa') / HIRO_REVISION / 'src'
    frozen = Path('.local/task275-rust') / ('hiro-gakumas-hero-' + HIRO_REVISION) / 'src'
    entry = Path('hiro/exec/exam.py')
    exam = resolve_runtime_path(frozen / entry, installed / entry)
    if not exam.is_file():
        raise ImportError(f'Hiro {HIRO_REVISION} exam source unavailable: {exam}')
    source = exam.parents[2]
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
    root = hiro_source().parent
    installed = root / relative
    if installed.is_file():
        return installed
    # The verified local UI and CLI helpers are optional to the shared exam
    # runtime. Development can exercise the exact same files before packaging.
    reference = resolve_runtime_path(
        Path('.local/task600-nonbattle/hiro-offline-' + HIRO_REVISION[:8]) / relative,
        Path('data/idol_road/rust_gnrpa') / HIRO_REVISION / relative,
    )
    if not reference.is_file():
        raise FileNotFoundError(f'Hiro {HIRO_REVISION} route file is unavailable: {relative}')
    return reference


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
