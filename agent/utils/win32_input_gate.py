"""Fail-closed MaaPiCli process and elevation validation for Win32 input."""

from __future__ import annotations

import os
import ctypes
from ctypes import wintypes
from typing import Protocol
from pathlib import Path
from dataclasses import dataclass

_MAAPICLI_EXE = "maapicli.exe"
_PYTHON_EXE = "python.exe"
_GAKUMAS_EXE = "gakumas.exe"
_HIGH_INTEGRITY_RID = 0x3000

_TH32CS_SNAPPROCESS = 0x00000002
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_TOKEN_QUERY = 0x0008
_TOKEN_ELEVATION_CLASS = 20
_TOKEN_INTEGRITY_LEVEL_CLASS = 25
_ERROR_NO_MORE_FILES = 18
_ERROR_INSUFFICIENT_BUFFER = 122


class Win32InputGateError(RuntimeError):
    """Raised before AgentServer startup when MaaPiCli input is not proven safe."""


@dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    parent_pid: int
    image_name: str


@dataclass(frozen=True)
class TopLevelWindow:
    window_handle: int
    pid: int
    class_name: str
    title: str


@dataclass(frozen=True)
class ProcessTokenFacts:
    elevated: bool
    integrity_rid: int


@dataclass(frozen=True)
class InspectedProcess:
    pid: int
    image_path: str
    token: ProcessTokenFacts


@dataclass(frozen=True)
class MaaPiCliInputGateEvidence:
    python_pid: int
    maapicli_pid: int
    game_pid: int
    game_window_handle: int
    python_path: str
    maapicli_path: str
    game_path: str
    integrity_rid: int


class ProcessInspector(Protocol):
    def snapshot(self) -> tuple[ProcessSnapshot, ...]: ...

    def visible_top_level_windows(self) -> tuple[TopLevelWindow, ...]: ...

    def inspect(self, pid: int) -> InspectedProcess: ...


class _ProcessEntry32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


class _TokenElevation(ctypes.Structure):
    _fields_ = [("TokenIsElevated", wintypes.DWORD)]


class _SidAndAttributes(ctypes.Structure):
    _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]


class _TokenMandatoryLabel(ctypes.Structure):
    _fields_ = [("Label", _SidAndAttributes)]


class WindowsProcessInspector:
    """Read only the Win32 process, image-path, and token facts used by the gate."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise Win32InputGateError("Win32 process inspection is unavailable on this platform")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._enum_windows_callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self._kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        self._kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        self._kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W)]
        self._kernel32.Process32FirstW.restype = wintypes.BOOL
        self._kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W)]
        self._kernel32.Process32NextW.restype = wintypes.BOOL
        self._kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

        self._user32.EnumWindows.argtypes = [self._enum_windows_callback, wintypes.LPARAM]
        self._user32.EnumWindows.restype = wintypes.BOOL
        self._user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self._user32.IsWindowVisible.restype = wintypes.BOOL
        self._user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self._user32.GetClassNameW.restype = ctypes.c_int
        self._user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self._user32.GetWindowTextLengthW.restype = ctypes.c_int
        self._user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self._user32.GetWindowTextW.restype = ctypes.c_int
        self._user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self._user32.GetWindowThreadProcessId.restype = wintypes.DWORD

        self._advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        self._advapi32.OpenProcessToken.restype = wintypes.BOOL
        self._advapi32.GetTokenInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._advapi32.GetTokenInformation.restype = wintypes.BOOL
        self._advapi32.IsValidSid.argtypes = [wintypes.LPVOID]
        self._advapi32.IsValidSid.restype = wintypes.BOOL
        self._advapi32.GetSidSubAuthorityCount.argtypes = [wintypes.LPVOID]
        self._advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(wintypes.BYTE)
        self._advapi32.GetSidSubAuthority.argtypes = [wintypes.LPVOID, wintypes.DWORD]
        self._advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)

    @staticmethod
    def _win32_error(operation: str, pid: int | None = None) -> Win32InputGateError:
        error = ctypes.get_last_error()
        target = "" if pid is None else f" for pid {pid}"
        return Win32InputGateError(f"{operation}{target} failed with Win32 error {error}")

    def snapshot(self) -> tuple[ProcessSnapshot, ...]:
        handle = self._kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
        invalid_handle = ctypes.c_void_p(-1).value
        if not handle or handle == invalid_handle:
            raise self._win32_error("CreateToolhelp32Snapshot")
        entries: list[ProcessSnapshot] = []
        try:
            entry = _ProcessEntry32W()
            entry.dwSize = ctypes.sizeof(entry)
            ctypes.set_last_error(0)
            if not self._kernel32.Process32FirstW(handle, ctypes.byref(entry)):
                raise self._win32_error("Process32FirstW")
            while True:
                entries.append(
                    ProcessSnapshot(
                        pid=int(entry.th32ProcessID),
                        parent_pid=int(entry.th32ParentProcessID),
                        image_name=str(entry.szExeFile),
                    )
                )
                entry.dwSize = ctypes.sizeof(entry)
                ctypes.set_last_error(0)
                if self._kernel32.Process32NextW(handle, ctypes.byref(entry)):
                    continue
                if ctypes.get_last_error() != _ERROR_NO_MORE_FILES:
                    raise self._win32_error("Process32NextW")
                break
        finally:
            self._kernel32.CloseHandle(handle)
        return tuple(entries)

    def visible_top_level_windows(self) -> tuple[TopLevelWindow, ...]:
        windows: list[TopLevelWindow] = []
        errors: list[Win32InputGateError] = []

        def visit(window_handle: int, _context: int) -> bool:
            if not self._user32.IsWindowVisible(window_handle):
                return True
            class_name = ctypes.create_unicode_buffer(256)
            ctypes.set_last_error(0)
            if self._user32.GetClassNameW(window_handle, class_name, len(class_name)) <= 0:
                errors.append(self._win32_error("GetClassNameW"))
                return False
            ctypes.set_last_error(0)
            title_length = self._user32.GetWindowTextLengthW(window_handle)
            if title_length < 0 or (title_length == 0 and ctypes.get_last_error() != 0):
                errors.append(self._win32_error("GetWindowTextLengthW"))
                return False
            title = ctypes.create_unicode_buffer(title_length + 1)
            ctypes.set_last_error(0)
            copied = self._user32.GetWindowTextW(window_handle, title, len(title))
            if copied == 0 and ctypes.get_last_error() != 0:
                errors.append(self._win32_error("GetWindowTextW"))
                return False
            process_id = wintypes.DWORD()
            if self._user32.GetWindowThreadProcessId(window_handle, ctypes.byref(process_id)) == 0 or process_id.value == 0:
                errors.append(self._win32_error("GetWindowThreadProcessId"))
                return False
            windows.append(
                TopLevelWindow(
                    window_handle=int(window_handle),
                    pid=int(process_id.value),
                    class_name=class_name.value,
                    title=title.value,
                )
            )
            return True

        callback = self._enum_windows_callback(visit)
        ctypes.set_last_error(0)
        if not self._user32.EnumWindows(callback, 0):
            if errors:
                raise errors[0]
            raise self._win32_error("EnumWindows")
        if errors:
            raise errors[0]
        return tuple(windows)

    def _token_facts(self, process_handle: wintypes.HANDLE, pid: int) -> ProcessTokenFacts:
        token_handle = wintypes.HANDLE()
        if not self._advapi32.OpenProcessToken(process_handle, _TOKEN_QUERY, ctypes.byref(token_handle)):
            raise self._win32_error("OpenProcessToken", pid)
        try:
            elevation = _TokenElevation()
            returned = wintypes.DWORD()
            if not self._advapi32.GetTokenInformation(
                token_handle,
                _TOKEN_ELEVATION_CLASS,
                ctypes.byref(elevation),
                ctypes.sizeof(elevation),
                ctypes.byref(returned),
            ):
                raise self._win32_error("GetTokenInformation(TokenElevation)", pid)

            required = wintypes.DWORD()
            ctypes.set_last_error(0)
            first = self._advapi32.GetTokenInformation(
                token_handle,
                _TOKEN_INTEGRITY_LEVEL_CLASS,
                None,
                0,
                ctypes.byref(required),
            )
            if first or ctypes.get_last_error() != _ERROR_INSUFFICIENT_BUFFER or required.value == 0:
                raise self._win32_error("GetTokenInformation(TokenIntegrityLevel size)", pid)
            buffer = ctypes.create_string_buffer(required.value)
            if not self._advapi32.GetTokenInformation(
                token_handle,
                _TOKEN_INTEGRITY_LEVEL_CLASS,
                buffer,
                required.value,
                ctypes.byref(required),
            ):
                raise self._win32_error("GetTokenInformation(TokenIntegrityLevel)", pid)
            mandatory_label = ctypes.cast(buffer, ctypes.POINTER(_TokenMandatoryLabel)).contents
            sid = mandatory_label.Label.Sid
            if not sid or not self._advapi32.IsValidSid(sid):
                raise Win32InputGateError(f"TokenIntegrityLevel for pid {pid} returned an invalid SID")
            authority_count = self._advapi32.GetSidSubAuthorityCount(sid)
            if not authority_count or authority_count.contents.value == 0:
                raise Win32InputGateError(f"TokenIntegrityLevel for pid {pid} returned no RID")
            integrity = self._advapi32.GetSidSubAuthority(sid, authority_count.contents.value - 1)
            if not integrity:
                raise self._win32_error("GetSidSubAuthority", pid)
            return ProcessTokenFacts(bool(elevation.TokenIsElevated), int(integrity.contents.value))
        finally:
            self._kernel32.CloseHandle(token_handle)

    def inspect(self, pid: int) -> InspectedProcess:
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise Win32InputGateError(f"invalid process id: {pid!r}")
        process_handle = self._kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not process_handle:
            raise self._win32_error("OpenProcess", pid)
        try:
            capacity = wintypes.DWORD(32768)
            image_path = ctypes.create_unicode_buffer(capacity.value)
            if not self._kernel32.QueryFullProcessImageNameW(
                process_handle,
                0,
                image_path,
                ctypes.byref(capacity),
            ):
                raise self._win32_error("QueryFullProcessImageNameW", pid)
            token = self._token_facts(process_handle, pid)
            return InspectedProcess(pid=pid, image_path=image_path.value, token=token)
        finally:
            self._kernel32.CloseHandle(process_handle)


def _canonical_path(path: str | os.PathLike[str], *, label: str) -> str:
    raw = os.fspath(path)
    if not raw or not Path(raw).is_absolute():
        raise Win32InputGateError(f"{label} path is missing or not absolute")
    try:
        return os.path.normcase(os.path.realpath(raw))
    except OSError as error:
        raise Win32InputGateError(f"{label} path cannot be resolved: {error}") from error


def _require_process_path(process: InspectedProcess, expected: Path, *, label: str) -> str:
    if process.pid <= 0:
        raise Win32InputGateError(f"{label} process identity is invalid")
    actual_path = _canonical_path(process.image_path, label=label)
    expected_path = _canonical_path(expected, label=f"expected {label}")
    if actual_path != expected_path:
        raise Win32InputGateError(f"{label} path does not match the active package")
    return actual_path


def _require_token(process: InspectedProcess, *, label: str) -> None:
    token = process.token
    if not isinstance(token.elevated, bool) or not token.elevated:
        raise Win32InputGateError(f"{label} TokenElevated is not true")
    if not isinstance(token.integrity_rid, int) or isinstance(token.integrity_rid, bool):
        raise Win32InputGateError(f"{label} integrity RID is unknown")
    if token.integrity_rid != _HIGH_INTEGRITY_RID:
        raise Win32InputGateError(f"{label} integrity RID is not High (0x{_HIGH_INTEGRITY_RID:04x})")


def validate_maapicli_win32_input_chain(
    inspector: ProcessInspector,
    *,
    current_pid: int,
    package_root: Path,
) -> MaaPiCliInputGateEvidence | None:
    """Validate one direct MaaPiCli -> Python Agent chain, or skip a non-CLI parent."""

    snapshots = inspector.snapshot()
    by_pid: dict[int, ProcessSnapshot] = {}
    for item in snapshots:
        if item.pid in by_pid:
            raise Win32InputGateError(f"process snapshot contains duplicate pid {item.pid}")
        by_pid[item.pid] = item
    current = by_pid.get(current_pid)
    if current is None:
        raise Win32InputGateError("current Python process is absent from the process snapshot")
    parent = by_pid.get(current.parent_pid)
    if parent is None:
        raise Win32InputGateError("current Python parent process is unknown")
    if parent.image_name.casefold() != _MAAPICLI_EXE:
        return None
    if current.image_name.casefold() != _PYTHON_EXE:
        raise Win32InputGateError("MaaPiCli child is not the package Python executable")
    if current.pid == parent.pid:
        raise Win32InputGateError("MaaPiCli parent-child relationship is invalid")

    games = tuple(item for item in snapshots if item.image_name.casefold() == _GAKUMAS_EXE)
    if len(games) != 1:
        raise Win32InputGateError(f"expected exactly one gakumas.exe process, found {len(games)}")
    game = games[0]
    if game.pid in {current.pid, parent.pid}:
        raise Win32InputGateError("gakumas.exe process relationship is invalid")

    game_windows = tuple(
        window
        for window in inspector.visible_top_level_windows()
        if window.class_name == "UnityWndClass" and "gakumas" in window.title.casefold()
    )
    if len(game_windows) != 1:
        raise Win32InputGateError(
            "expected exactly one visible UnityWndClass window with a gakumas title, "
            f"found {len(game_windows)}"
        )
    game_window = game_windows[0]
    if game_window.pid != game.pid:
        raise Win32InputGateError("gakumas window pid does not match the unique gakumas.exe process")

    inspected_python = inspector.inspect(current.pid)
    inspected_parent = inspector.inspect(parent.pid)
    inspected_game = inspector.inspect(game.pid)
    if inspected_python.pid != current.pid or inspected_parent.pid != parent.pid or inspected_game.pid != game.pid:
        raise Win32InputGateError("process inspection returned a mismatched pid")

    root = Path(_canonical_path(package_root, label="package root"))
    python_path = _require_process_path(inspected_python, root / "python" / "python.exe", label="Python")
    maapicli_path = _require_process_path(
        inspected_parent,
        root / "MaaPiCli.exe",
        label="MaaPiCli",
    )
    game_path = _canonical_path(inspected_game.image_path, label="gakumas.exe")
    if Path(game_path).name.casefold() != _GAKUMAS_EXE:
        raise Win32InputGateError("gakumas.exe snapshot and image path disagree")

    _require_token(inspected_python, label="Python")
    _require_token(inspected_parent, label="MaaPiCli")
    _require_token(inspected_game, label="gakumas.exe")
    integrity_rids = {
        inspected_python.token.integrity_rid,
        inspected_parent.token.integrity_rid,
        inspected_game.token.integrity_rid,
    }
    if integrity_rids != {_HIGH_INTEGRITY_RID}:
        raise Win32InputGateError("Python, MaaPiCli, and gakumas.exe do not share one High integrity RID")

    return MaaPiCliInputGateEvidence(
        python_pid=current.pid,
        maapicli_pid=parent.pid,
        game_pid=game.pid,
        game_window_handle=game_window.window_handle,
        python_path=python_path,
        maapicli_path=maapicli_path,
        game_path=game_path,
        integrity_rid=_HIGH_INTEGRITY_RID,
    )


def enforce_maapicli_win32_input_gate() -> MaaPiCliInputGateEvidence | None:
    """Apply the gate to the installed Win32 agent before AgentServer startup."""

    if os.name != "nt":
        return None
    package_root = Path(__file__).resolve().parents[2]
    return validate_maapicli_win32_input_chain(
        WindowsProcessInspector(),
        current_pid=os.getpid(),
        package_root=package_root,
    )
