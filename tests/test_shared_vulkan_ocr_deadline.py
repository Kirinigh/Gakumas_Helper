"""Use real worker pipes to verify the deadline on the shared OCR route."""

import sys
from time import sleep, monotonic
from types import SimpleNamespace
from threading import Event
from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from agent.live_decision import simd_ocr
from agent.live_decision.ocr_nodes import read_ocr_node

_READY = """print('{"ready": true}', flush=True)"""
FAULTS = {
    "missing-ready": "time.sleep(8)",
    "blocked-write": f"{_READY}; time.sleep(8)",
    "missing-reply": f"{_READY}; sys.stdin.readline(); time.sleep(8)",
}
NODE_DATA = {"recognition": {"type": "OCR", "param": {"expected": ".+"}}}


def faulty_vulkan(fault):
    return simd_ocr.VulkanOcr((sys.executable, "-u", "-c", f"import sys,time; {FAULTS[fault]}"))


def test_shared_factory_reuses_one_bounded_gpu_worker(monkeypatch, tmp_path):
    helper = tmp_path / "GkhSimdOcr.dll"
    helper.touch()
    launch = Mock(side_effect=AssertionError("worker should start only when it has a frame"))
    monkeypatch.setenv("GKH_SIMD_OCR_MODEL", "small")
    monkeypatch.setattr(simd_ocr, "resolve_runtime_path", lambda *_: helper)
    monkeypatch.setattr(simd_ocr.shutil, "which", lambda _: sys.executable)
    monkeypatch.setattr(simd_ocr.subprocess, "Popen", launch)
    simd_ocr._configured_simd_ocr.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            workers = list(pool.map(simd_ocr.configured_simd_ocr, ["vulkan"] * 4))
        assert isinstance(workers[0], simd_ocr.VulkanOcr)
        assert all(worker is workers[0] for worker in workers)
        assert type(simd_ocr.configured_simd_ocr("cpu")) is simd_ocr.SimdOcr
        launch.assert_not_called()
    finally:
        simd_ocr._configured_simd_ocr.cache_clear()


@pytest.mark.parametrize("fault", FAULTS)
def test_shared_route_bounds_cold_faults_without_repeating_the_foreground_wait(monkeypatch, fault):
    gpu = faulty_vulkan(fault)
    gpu._start_timeout = gpu._reply_timeout = 0.35
    launched = Mock(wraps=gpu._start)
    gpu._start = launched
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(2)]
    row = SimpleNamespace(text="verified", score=1.0, box=(0, 0, 15, 15))
    cpu = SimpleNamespace(available=True, read=Mock(return_value=(row,)))
    workers = {"vulkan": gpu, "cpu": cpu}
    monkeypatch.setenv("GKH_SIMD_OCR", "auto")
    monkeypatch.setattr(simd_ocr, "VULKAN_COLD_WAIT_SECONDS", 0.15)
    monkeypatch.setattr(simd_ocr, "configured_simd_ocr", workers.get)
    context = SimpleNamespace(get_node_data=lambda _: NODE_DATA)

    try:
        started = monotonic()
        assert read_ocr_node(context, frames[0], "StructuredOCR", prefer_gpu=True).rows == (row,)
        first = monotonic() - started
        assert 0.12 <= first <= 0.55, (fault, first)
        pending = gpu._pending
        assert gpu.available and pending is not None
        assert cpu.read.call_args.args[0] is frames[0]

        started = monotonic()
        assert read_ocr_node(context, frames[1], "StructuredOCR", prefer_gpu=True).rows == (row,)
        assert monotonic() - started < 0.25
        assert cpu.read.call_args.args[0] is frames[1]
        with pytest.raises(RuntimeError, match="timed out|did not reply"):
            pending.result(timeout=1.2)
        assert not gpu.available
        starts = launched.call_count
        assert 1 <= starts <= 2
        assert read_ocr_node(context, frames[1], "StructuredOCR", prefer_gpu=True).rows == (row,)
        assert launched.call_count == starts and cpu.read.call_count == 3
    finally:
        gpu.close()


def test_a_successful_cold_exchange_can_warm_vulkan_without_charging_the_caller(monkeypatch):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    # Make completion deterministic without a machine-dependent performance bound.
    release = Event()
    original = simd_ocr.SimdOcr.read
    row = SimpleNamespace(text="vulkan", score=1.0)

    def delayed(worker, image, **params):
        assert image is frame
        if worker is gpu:
            assert release.wait(1)
            return (row,)
        return original(worker, image, **params)

    gpu = simd_ocr.VulkanOcr(())
    cpu_row = SimpleNamespace(text="cpu", score=1.0)
    cpu = SimpleNamespace(available=True, read=Mock(return_value=(cpu_row,)))
    monkeypatch.setenv("GKH_SIMD_OCR", "auto")
    monkeypatch.setattr(simd_ocr, "configured_simd_ocr", {"vulkan": gpu, "cpu": cpu}.get)
    monkeypatch.setattr(simd_ocr.SimdOcr, "read", delayed)
    try:
        with simd_ocr.without_vulkan_cold_wait():
            assert simd_ocr.read_optional_simd_ocr(frame, {}, prefer_gpu=True) == (cpu_row,)
        release.set()
        deadline = monotonic() + 0.5
        while not gpu._warm and monotonic() < deadline:
            sleep(0.005)
        assert simd_ocr.read_optional_simd_ocr(frame, {}, prefer_gpu=True) == (row,)
        cpu.read.assert_called_once()
    finally:
        release.set()
        gpu.close()


def test_slow_first_frame_switches_to_cpu_and_can_enable_gpu_for_later_frames(monkeypatch):
    gpu = simd_ocr.VulkanOcr(())
    entered = Event()
    release = Event()
    calls = []
    first = np.zeros((32, 32, 3), dtype=np.uint8)
    later = np.ones_like(first)
    first_gpu_row = SimpleNamespace(text="first", score=1.0)
    real_row = SimpleNamespace(text="frame", score=1.0)
    cpu_row = SimpleNamespace(text="cpu", score=1.0)
    cpu = SimpleNamespace(available=True, read=Mock(return_value=(cpu_row,)))
    monkeypatch.setenv("GKH_SIMD_OCR", "auto")
    monkeypatch.setattr(simd_ocr, "VULKAN_COLD_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(simd_ocr, "configured_simd_ocr", {"vulkan": gpu, "cpu": cpu}.get)

    def read(worker, image, **params):
        assert worker is gpu
        if image is later:
            return (real_row,)
        assert image is first
        calls.append(params)
        entered.set()
        assert release.wait(2)
        return (first_gpu_row,)

    monkeypatch.setattr(simd_ocr.SimdOcr, "read", read)
    try:
        assert gpu.process is None and gpu._pending is None and calls == []
        assert simd_ocr.read_optional_simd_ocr(first, {}, prefer_gpu=True) == (cpu_row,)
        assert entered.wait(1)
        pending = gpu._pending
        assert simd_ocr.read_optional_simd_ocr(later, {}, prefer_gpu=True) == (cpu_row,)
        assert gpu.available and not pending.done()
        assert cpu.read.call_args_list[0].args[0] is first
        assert cpu.read.call_args_list[1].args[0] is later

        release.set()
        assert pending.result(timeout=1) == (first_gpu_row,)
        assert calls == [{}]
        assert simd_ocr.read_optional_simd_ocr(later, {}, prefer_gpu=True) == (real_row,)
    finally:
        release.set()
        gpu.close()


@pytest.mark.parametrize("mode", ("cpu", "maa"))
def test_cold_gpu_is_not_started_when_disabled(monkeypatch, mode):
    configure = Mock(return_value=None)
    monkeypatch.setenv("GKH_SIMD_OCR", mode)
    monkeypatch.setattr(simd_ocr, "configured_simd_ocr", configure)
    assert simd_ocr.read_optional_simd_ocr(None, {}, prefer_gpu=True) == ()
    assert all(call.args == ("cpu",) for call in configure.call_args_list)


def test_first_reply_can_use_the_longer_background_budget(monkeypatch):
    command = """import sys,time
print('{"ready": true}', flush=True)
for index, request in enumerate(sys.stdin):
    if index == 0:
        time.sleep(3.1)
    print('{"lines": []}', flush=True)
"""
    gpu = simd_ocr.VulkanOcr((sys.executable, "-u", "-c", command))
    monkeypatch.setattr(simd_ocr, "VULKAN_COLD_WAIT_SECONDS", 0.05)
    try:
        assert gpu.read(np.zeros((32, 32, 3), dtype=np.uint8)) is None
        assert gpu._pending.result(timeout=5.5) == ()
        assert gpu._warm and gpu.available
    finally:
        gpu.close()


def test_cold_transport_restarts_once_and_retries_the_actual_frame(tmp_path):
    marker = tmp_path / "first-start"
    command = """import json,sys
from pathlib import Path
marker = Path(sys.argv[1])
if not marker.exists():
    marker.touch()
    sys.exit(1)
print('{"ready": true}', flush=True)
for request in sys.stdin:
    parsed = json.loads(request)
    assert (parsed['width'], parsed['height']) == (32, 32)
    print('{"lines": [{"Text": "verified", "Score": 1, "Left": 1, "Top": 2, "Right": 12, "Bottom": 14}]}', flush=True)
"""
    gpu = simd_ocr.VulkanOcr((sys.executable, "-u", "-c", command, str(marker)))
    gpu._start = Mock(wraps=gpu._start)
    try:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with simd_ocr.without_vulkan_cold_wait():
            assert gpu.read(frame) is None
        pending = gpu._pending
        rows = gpu.read(frame) if pending is None else pending.result(timeout=2)
        assert [row.text for row in rows] == ["verified"]
        assert gpu._start.call_count == 2 and gpu.available
        assert [row.text for row in gpu.read(frame)] == ["verified"]
        assert gpu._start.call_count == 2
    finally:
        gpu.close()
