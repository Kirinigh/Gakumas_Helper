"""Reuse decoded evidence only within one recovery call, never input authority."""
from __future__ import annotations

import json
import mmap
from copy import deepcopy
from pathlib import Path
from functools import wraps
from contextvars import ContextVar

from .hif_live_io import freeze_record_history

_READS = ContextVar('hif_recovery_report_reads', default=None)
_CHAINS = ContextVar('hif_recovery_control_chains', default=None)


def recovery_report_scope(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        if _READS.get() is not None:
            return function(*args, **kwargs)
        token = _READS.set({})
        chain_token = _CHAINS.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _CHAINS.reset(chain_token)
            _READS.reset(token)
    return scoped


def read_recovery_report(path):
    """Return independent current fields and immutable original control history.

    The scope ends on every return/exception from recovery. Replaced files are
    reread, and callers still run all identity, budget and receipt validations.
    This is not used for the current owner record or OS/input permissions.
    """
    path = Path(path)
    cache = _READS.get()
    if cache is None:
        return json.loads(path.read_bytes())
    stamp = path.stat()
    key = (path.resolve(), stamp.st_ino, stamp.st_size, stamp.st_mtime_ns)
    if key not in cache:
        from route.cultivation_factory import _decode_private_control_history

        row = _decode_private_control_history(path.read_bytes())
        if 'automatic_battle' in row:
            row['automatic_battle'] = freeze_record_history(row['automatic_battle'])
        # Drop an older version of this same path instead of accumulating it.
        for old in tuple(cache):
            if old[0] == key[0]:
                del cache[old]
        cache[key] = row
    original = cache[key]
    return {name: value if name == 'automatic_battle' else deepcopy(value)
            for name, value in original.items()}


def recovery_report_contains_handoff(path, handoff_id):
    """Exclude unrelated histories without decoding their large control trees.

    A match is only a candidate. The original full loader still validates its
    top-level identity, complete chain and budgets before any recovery.
    """
    path = Path(path)
    before = path.stat()
    cache = _READS.get()
    key = ('handoff_candidate', handoff_id, path.resolve(), before.st_ino, before.st_size, before.st_mtime_ns)
    if cache is not None and key in cache:
        return cache[key]
    needles = {json.dumps(handoff_id, ensure_ascii=escaped).encode('utf-8') for escaped in (True, False)}
    with path.open('rb') as stream:
        if before.st_size:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
                found = any(raw.find(needle) >= 0 for needle in needles)
        else:
            found = False
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('recovery report changed while selecting its handoff')
    if cache is not None:
        cache[key] = found
    return found


def matching_recovery_reports(runtime, handoff_id):
    return [path for path in sorted(Path(runtime).glob('result-*-live*.json'))
        if recovery_report_contains_handoff(path, handoff_id)]


def recovery_control_chain(runtime, bound, loader, *, sources=None, variant=None):
    """Reuse the exact checked historical chain within this recovery call.

    Rescan all source paths and versions on every use. This contains no current
    owner, deadline decision, retry grant or input authority; those are still
    checked by each caller. No result survives the outer recovery invocation.
    """
    sources = sources or (lambda: sorted(Path(runtime).glob('result-*-live*.json')))
    paths = sources()
    cache = _CHAINS.get()
    if cache is None:
        return loader(paths)
    versions = tuple((path.resolve(), stat.st_ino, stat.st_size, stat.st_mtime_ns)
        for path in paths for stat in (path.stat(),))
    key = (Path(runtime).resolve(), tuple(sorted(bound.items())), variant, versions)
    if key not in cache:
        ordered = loader(paths)
        after = tuple((path.resolve(), stat.st_ino, stat.st_size, stat.st_mtime_ns)
            for path in sources() for stat in (path.stat(),))
        if versions != after:
            raise ValueError('HIF automatic control sources changed during validation')
        cache.clear()
        cache[key] = ordered
    return [{name: value if name == 'automatic_battle' else deepcopy(value)
             for name, value in row.items()} for row in cache[key]]


def read_recovery_business_report(path):
    """Read Native business receipts without decoding unrelated control prefixes.

    The compact writer places automatic_battle immediately before
    total_input_count. Both delimiters must be unique, and the remaining JSON
    must be complete and bounded. This result deliberately carries no execution
    control, budget or input authority; their current owner must check them.
    """
    path = Path(path)
    before = path.stat()
    with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        first, next_field = b',"automatic_battle":', b',"total_input_count":'
        start = raw.find(first)
        end = raw.find(next_field, start) if start >= 0 else -1
        if (start > 0 and end > start and raw.find(first, start+1) < 0
                and raw.find(next_field, end+1) < 0 and start+len(raw)-end < 8_000_000):
            body = raw[:start]+raw[end:]
        elif len(raw) < 8_000_000:
            body = raw[:]
        else:
            raise ValueError('native business record is not a bounded compact projection')
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('native business record changed while reading')
    result = json.loads(body)
    if not isinstance(result, dict) or path.name != f"result-{result.get('id')}.json":
        raise ValueError('native business record identity differs from its path')
    result.pop('automatic_battle', None)
    return result
