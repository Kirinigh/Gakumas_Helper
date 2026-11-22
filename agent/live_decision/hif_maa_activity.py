"""Read-only activity check for the existing HIF acceptance Maa launcher.

An open GUI is not an executing queue. Unknown GUI/log state still refuses
concurrent use; this does not grant input, terminate a GUI or replace the
original task, process-instance, elevation and fresh-frame checks.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from datetime import datetime


def _timestamp(line):
    try:
        return datetime.strptime(line[1:24], '%Y-%m-%d %H:%M:%S.%f')
    except ValueError as error:
        raise ValueError('hif_maa_activity_log_timestamp_unknown') from error


def gui_idle_evidence(gui_text, native_text, *, pid, created_at):
    """Only an ended queue and its inactive native controller establish idle."""
    instances = {}
    for line in gui_text.splitlines():
        if not line.startswith('[') or '[inst=' not in line:
            continue
        if _timestamp(line) < created_at:
            continue
        instance = re.search(r'\[inst=([^]]+)\]', line).group(1)
        # These are whole-queue terminal messages. An individual task failure
        # is not enough: the GUI may still be starting the next queued task.
        if '任务已全部完成！' in line or '任务运行失败！' in line:
            instances[instance] = ('complete', _timestamp(line))
        elif '[src=Worker]' in line or '开始任务：' in line:
            instances[instance] = ('busy', _timestamp(line))
    if not instances or any(state != 'complete' for state, _ in instances.values()):
        raise ValueError('hif_maa_gui_queue_active_or_unknown')

    tasks, controllers = {}, {}
    marker = f'[Px{pid}]'
    for line in native_text.splitlines():
        if marker not in line or _timestamp(line) < created_at:
            continue
        event = re.search(r'\[(?:msg|message)=((?:Tasker.Task|Controller.Action)\.[^]]+)\]', line)
        if not event:
            continue
        identity = re.search(r'"uuid":"([^"]+)"', line)
        if identity is None:
            raise ValueError('hif_maa_gui_native_identity_unknown')
        key, name = identity.group(1), event.group(1)
        if name.startswith('Tasker.Task.'):
            # A failed task has also stopped. Only the independent inactive
            # controller event below establishes that it released input.
            tasks[key] = name in ('Tasker.Task.Succeeded', 'Tasker.Task.Failed')
        else:
            controllers[key] = (name == 'Controller.Action.Succeeded'
                                and '"action":"inactive"' in line)
    if (not tasks or tasks.keys() != controllers.keys()
            or not all(tasks.values()) or not all(controllers.values())):
        raise ValueError('hif_maa_gui_native_task_active_or_unknown')
    return dict(pid=pid, state='idle', queues=list(instances),
                completed_at=max(at for _, at in instances.values()).isoformat(),
                controllers=list(controllers))


def _other_maa(processes, own_pid):
    ancestors, current = set(), own_pid
    for _ in range(8):
        if current not in processes or current in ancestors:
            break
        ancestors.add(current)
        current = processes[current][0]
    others = {pid: row for pid, row in processes.items()
              if row[1].lower() in ('maapicli.exe', 'maagakumasu.exe')
              and pid not in ancestors}
    if any(name.lower() == 'maapicli.exe' for _, name in others.values()):
        raise ValueError('hif_maa_other_cli_running')
    return others


def _read_gui(instance, pid):
    root = Path(instance['executable']).parent
    if Path(instance['executable']).name.lower() != 'maagakumasu.exe':
        raise ValueError('hif_maa_gui_executable_changed')
    created_at = datetime.fromtimestamp(instance['created_at_100ns'] / 10_000_000 - 11_644_473_600)
    # The installed client's dated log is authoritative; missing/rotated
    # lifecycle evidence is unknown, never a synthetic idle state.
    log = root / 'logs' / f'log-{datetime.now():%Y%m%d}.log'
    native = root / 'debug' / 'maafw.log'
    before = [(path.stat().st_size, path.stat().st_mtime_ns) for path in (log, native)]
    evidence = gui_idle_evidence(log.read_text(encoding='utf-8-sig'),
                                native.read_text(encoding='utf-8-sig'),
                                pid=pid, created_at=created_at)
    after = [(path.stat().st_size, path.stat().st_mtime_ns) for path in (log, native)]
    if before != after:
        raise ValueError('hif_maa_gui_activity_changed_during_check')
    return dict(evidence, executable=instance['executable'],
                created_at_100ns=instance['created_at_100ns'], log=str(log),
                native_log=str(native), native_bytes=after[1][0])


def require_idle_maa():
    """Re-read real task activity; no saved input permission or GUI operation."""
    from route.bootstrap import _process_instance
    from utils.maa_input import _process_entries

    processes = _process_entries()
    if processes.get(4, (None, ''))[1].lower() != 'system':
        raise ValueError('hif_maa_system_process_view_unavailable')
    others = _other_maa(processes, os.getpid())
    evidence = []
    for pid in others:
        instance = _process_instance(pid)
        evidence.append(_read_gui(instance, pid))
        if _process_instance(pid) != instance:
            raise ValueError('hif_maa_gui_process_instance_changed')
    if _other_maa(_process_entries(), os.getpid()) != others:
        raise ValueError('hif_maa_activity_processes_changed')
    return evidence
