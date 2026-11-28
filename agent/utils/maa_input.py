"""Read Windows process tokens before sanctioned Maa input; no input implementation."""

import os
import ctypes
from pathlib import Path

try:
    from agent.live_decision.layout import calibrated_live_client_size
except ModuleNotFoundError:
    from live_decision.layout import calibrated_live_client_size


class MaaInputEndpointError(RuntimeError):
    pass
def process_integrity_rid(pid: int) -> int:
    from ctypes import wintypes

    class SidAndAttributes(ctypes.Structure):
        _fields_ = [("sid", wintypes.LPVOID), ("attributes", wintypes.DWORD)]

    class TokenMandatoryLabel(ctypes.Structure):
        _fields_ = [("label", SidAndAttributes)]

    process_query_limited_information = 0x1000
    token_query = 0x0008
    token_integrity_level = 25
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.GetSidSubAuthorityCount.argtypes = [wintypes.LPVOID]
    advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
    advapi32.GetSidSubAuthority.argtypes = [wintypes.LPVOID, wintypes.DWORD]
    advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)

    process = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not process:
        raise MaaInputEndpointError(f"could not open process for token inspection: pid={pid}")
    token = wintypes.HANDLE()
    try:
        if not advapi32.OpenProcessToken(process, token_query, ctypes.byref(token)):
            raise MaaInputEndpointError(f"could not open process token: pid={pid}")
        needed = wintypes.DWORD()
        advapi32.GetTokenInformation(
            token,
            token_integrity_level,
            None,
            0,
            ctypes.byref(needed),
        )
        if needed.value <= 0:
            raise MaaInputEndpointError(f"could not size integrity token: pid={pid}")
        buffer = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(
            token,
            token_integrity_level,
            buffer,
            needed,
            ctypes.byref(needed),
        ):
            raise MaaInputEndpointError(f"could not read integrity token: pid={pid}")
        mandatory_label = ctypes.cast(
            buffer,
            ctypes.POINTER(TokenMandatoryLabel),
        ).contents
        sid_pointer = mandatory_label.label.sid
        if not sid_pointer:
            raise MaaInputEndpointError(f"integrity token contains no SID: pid={pid}")
        count_pointer = advapi32.GetSidSubAuthorityCount(sid_pointer)
        if not count_pointer or count_pointer.contents.value <= 0:
            raise MaaInputEndpointError(f"could not read integrity SID length: pid={pid}")
        rid_pointer = advapi32.GetSidSubAuthority(
            sid_pointer,
            count_pointer.contents.value - 1,
        )
        if not rid_pointer:
            raise MaaInputEndpointError(f"could not read integrity RID: pid={pid}")
        return int(rid_pointer.contents.value)
    finally:
        if token:
            kernel32.CloseHandle(token)
        kernel32.CloseHandle(process)


def _process_entries():
    """Read process metadata only, including Windows venv redirector parents."""
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD),
                    ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                    ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent", wintypes.DWORD), ("priority", wintypes.LONG),
                    ("flags", wintypes.DWORD), ("executable", wintypes.WCHAR * 260)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise MaaInputEndpointError("input process ancestry is unavailable")
    entry, processes = ProcessEntry(), {}
    entry.size = ctypes.sizeof(entry)
    try:
        if not kernel.Process32FirstW(snapshot, ctypes.byref(entry)):
            raise MaaInputEndpointError("input process ancestry is empty")
        while True:
            processes[int(entry.pid)] = (int(entry.parent), entry.executable)
            if not kernel.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel.CloseHandle(snapshot)
    return processes


def _parent_process_ids():
    return {pid: row[0] for pid, row in _process_entries().items()}


def verify_maa_input_endpoints(controller_info, *, expected_client_size=(720, 1280), require_calibrated_size=True):
    """Bind the real Win32 controller window, Maa host and Agent to high tokens."""
    from ctypes import wintypes

    # Callers with normalized capture validate its geometry independently.
    profile = calibrated_live_client_size(expected_client_size) if require_calibrated_size else None
    if require_calibrated_size and profile is None:
        raise MaaInputEndpointError("unsupported observed Maa client size")
    if (not isinstance(expected_client_size, tuple) or len(expected_client_size) != 2
            or any(type(value) is not int or value <= 0 for value in expected_client_size)):
        raise MaaInputEndpointError("Maa input target size unavailable")
    if (os.name != "nt" or not isinstance(controller_info, dict)
            or controller_info.get("type") != "win32"
            or type(controller_info.get("hwnd")) is not int
            or controller_info["hwnd"] <= 0
            or controller_info.get("mouse_method") != 32):
        raise MaaInputEndpointError("sanctioned Win32 Maa input endpoint is unresolved")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    hwnd = controller_info["hwnd"]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    title, cls = ctypes.create_unicode_buffer(256), ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, title, 256)
    user32.GetClassNameW(hwnd, cls, 256)
    if title.value != "gakumas" or cls.value != "UnityWndClass":
        raise MaaInputEndpointError("Maa controller is not bound to the game window")
    game_pid, rect = wintypes.DWORD(), wintypes.RECT()
    if not user32.GetWindowThreadProcessId(hwnd, ctypes.byref(game_pid)):
        raise MaaInputEndpointError("game PID is unavailable")
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise MaaInputEndpointError("Maa input target size unavailable")
    actual_size = (rect.right-rect.left, rect.bottom-rect.top)
    if (min(actual_size) <= 0 or any(abs(v-old) > 1 for v,old in zip(actual_size, expected_client_size, strict=True))
            or (require_calibrated_size and calibrated_live_client_size(actual_size) != profile)):
        raise MaaInputEndpointError("Maa input target size changed")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                  wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    parents = _parent_process_ids()
    pid, chain = os.getpid(), []
    for _ in range(4):
        process = kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            raise MaaInputEndpointError("input ancestor process is unavailable")
        try:
            path, length = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
            if not kernel32.QueryFullProcessImageNameW(process, 0, path, ctypes.byref(length)):
                raise MaaInputEndpointError("input ancestor executable is unavailable")
            executable = Path(path.value).name.lower()
            chain.append({"pid": pid, "executable": path.value,
                          "integrity_rid": process_integrity_rid(pid)})
        finally:
            kernel32.CloseHandle(process)
        if executable in {"maapicli.exe", "maagakumasu.exe"}:
            break
        if executable not in {"python.exe", "pythonw.exe"} or pid not in parents:
            raise MaaInputEndpointError("input ancestor is not a Maa Python child: " + executable)
        pid = parents[pid]
    else:
        raise MaaInputEndpointError("Maa host is not in the bounded Agent ancestry")
    result = {"agent": chain[0], "maa": chain[-1],
              "game": {"pid": game_pid.value, "integrity_rid": process_integrity_rid(game_pid.value),
                       "observed_client_size": actual_size, "reference_client_size": expected_client_size,
                       "calibrated_client_size": profile}}
    result.update({f"input_child_{index}": row for index, row in enumerate(chain[1:-1], 1)})
    if any(row["integrity_rid"] != 0x3000 for row in result.values()):
        raise MaaInputEndpointError("Maa, Agent and game must each have high integrity")
    return result
