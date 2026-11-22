"""One trusted source tree for the existing local HIF runtime dependencies.

No path comes from a task, checkpoint, environment variable, or saved command.
The native readers live under agent. The agent package name used by the frozen
fast-policy business bridge is resolved from this same installed tree only.
"""
from __future__ import annotations

import sys
from pathlib import Path
from functools import wraps
from contextlib import contextmanager


def _source_root():
    return Path(__file__).resolve().parents[2]


def _verify_loaded(root, native_names):
    for name, module in tuple(sys.modules.items()):
        if name == 'agent' or name.startswith('agent.'):
            directory = root / 'agent'
        elif name == 'live_decision' or name.startswith('live_decision.'):
            directory = root / 'agent/live_decision'
        elif name in native_names:
            directory = root / 'agent/live_decision/hif_native'
        else:
            continue
        file = getattr(module, '__file__', None)
        paths = (file,) if file else tuple(getattr(module, '__path__', ()))
        if not paths or any(not Path(path).resolve().is_relative_to(directory) for path in paths):
            raise ValueError('HIF dependency is loaded from a different code source: ' + name)


@contextmanager
def hif_runtime_sources():
    """Resolve the frozen dual business namespace from this installed tree."""
    root = _source_root()
    native = root / 'agent/live_decision/hif_native'
    from .hif_native import RUNTIME_MODULES
    required = RUNTIME_MODULES
    if (not native.resolve().is_relative_to(root)
            or any(not (native / (name + '.py')).is_file() for name in required)):
        raise ValueError('HIF runtime source dependencies are missing from this code tree')
    names = {path.stem for path in native.glob('*.py')}
    _verify_loaded(root, names)
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(root))
        # This is the installed source root, never an external checkout.
        _verify_loaded(root, names)
        yield native
        _verify_loaded(root, names)
    finally:
        sys.path[:] = previous


def with_hif_runtime_sources(method):
    @wraps(method)
    def bound(*args, **kwargs):
        with hif_runtime_sources():
            return method(*args, **kwargs)
    return bound
