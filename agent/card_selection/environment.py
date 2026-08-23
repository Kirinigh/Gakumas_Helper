"""Windows-only DMM target validation shared by runtime and intake tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DmmWindowFacts:
    window_handle: int
    window_class: str
    process_name: str
    client_size: tuple[int, int]


def validate_dmm_window(window_handle: int, *, expected_size: tuple[int, int] = (720, 1280)) -> DmmWindowFacts:
    if os.name != "nt":
        raise RuntimeError("DMM window validation is Windows-only")
    if not isinstance(window_handle, int) or window_handle <= 0:
        raise ValueError("window handle must be a positive integer")

    import ctypes

    class ClientRect(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    user32 = ctypes.windll.user32
    if not user32.IsWindow(window_handle):
        raise RuntimeError("target window handle is not valid")
    class_name = ctypes.create_unicode_buffer(256)
    if user32.GetClassNameW(window_handle, class_name, len(class_name)) <= 0 or class_name.value != "UnityWndClass":
        raise RuntimeError(f"target window class is not approved: {class_name.value}")
    rect = ClientRect()
    if not user32.GetClientRect(window_handle, ctypes.byref(rect)):
        raise RuntimeError("target client rectangle cannot be read")
    client_size = (rect.right - rect.left, rect.bottom - rect.top)
    if client_size != expected_size:
        raise RuntimeError(f"target client size {client_size} does not match approved DMM size {expected_size}")

    process_id = ctypes.c_uint32()
    user32.GetWindowThreadProcessId(window_handle, ctypes.byref(process_id))
    process = ctypes.windll.kernel32.OpenProcess(0x1000, False, process_id.value)
    if not process:
        raise RuntimeError("target process identity cannot be opened")
    try:
        capacity = ctypes.c_uint32(32768)
        image_path = ctypes.create_unicode_buffer(capacity.value)
        if not ctypes.windll.kernel32.QueryFullProcessImageNameW(process, 0, image_path, ctypes.byref(capacity)):
            raise RuntimeError("target process identity cannot be read")
    finally:
        ctypes.windll.kernel32.CloseHandle(process)
    process_name = Path(image_path.value).name.lower()
    if process_name != "gakumas.exe":
        raise RuntimeError(f"target process is not approved: {process_name}")
    return DmmWindowFacts(window_handle, class_name.value, process_name, client_size)
