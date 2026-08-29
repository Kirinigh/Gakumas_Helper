"""Shared fail-closed privacy gate for public source and release trees."""

from __future__ import annotations

import io
import os
import re
import stat
import zipfile
from pathlib import Path
from collections.abc import Callable, Collection

TEXT_SUFFIXES = {
    ".bat",
    ".cfg",
    ".cmd",
    ".conf",
    ".csv",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".toml",
    ".ts",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ARCHIVE_SUFFIXES = {".jar", ".npz", ".whl", ".zip"}
FORBIDDEN_PRIVATE_NAMES = {
    ".env",
    ".env.local",
    ".netrc",
    ".pypirc",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}
FORBIDDEN_PRIVATE_SUFFIXES = {".bak", ".db", ".dmp", ".dump", ".log", ".sqlite", ".sqlite3", ".tmp"}
FORBIDDEN_PRIVATE_COMPONENTS = {
    ".local",
    "captures",
    "logs",
    "profiles",
    "runtime-data",
    "screenshots",
    "temp",
    "tmp",
    "user_data",
    "userdata",
}
SECRET_BYTE_PATTERNS = (
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(rb"Authorization\s*:\s*Bearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
)
EMAIL_PATTERN = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
BINARY_EMAIL_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9._%+-])[A-Za-z0-9][A-Za-z0-9._%+-]{2,}"
    r"@(?:[A-Za-z0-9-]{2,}\.)+[A-Za-z]{2,24}(?![A-Za-z0-9_.-])"
)
ASCII_TEXT_RUN_PATTERN = re.compile(rb"[\x20-\x7e]{6,}")
UTF16_LE_TEXT_RUN_PATTERN = re.compile(rb"(?:[\x20-\x7e]\x00){6,}")
UTF16_BE_TEXT_RUN_PATTERN = re.compile(rb"(?:\x00[\x20-\x7e]){6,}")
_MACOS_PROFILE_PREFIX = "/" + "Users/"
_LINUX_PROFILE_PREFIX = "/" + "home/"
_PRIVATE_SAMPLE_PREFIXES = ("custom_" + "positive", "private_" + "sample", "user_" + "capture")
_PRIVATE_SAMPLE_KEY_PATTERN = re.compile(
    rf'(?i)"(?:{"|".join(_PRIVATE_SAMPLE_PREFIXES)})[^"]*sha{"256"}"\s*:'
)
PERSONAL_TEXT_PATTERNS = (
    ("Windows user profile path", re.compile(r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\s]+")),
    ("macOS user profile path", re.compile(re.escape(_MACOS_PROFILE_PREFIX) + r"[^/\s]+")),
    ("Linux user profile path", re.compile(re.escape(_LINUX_PROFILE_PREFIX) + r"[^/\s]+")),
    (
        "private network address",
        re.compile(r"(?<!\d)(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d{1,3}\.\d{1,3}(?!\d)"),
    ),
    ("hardware address", re.compile(r"(?i)(?<![0-9a-f])(?:[0-9a-f]{2}:){5}[0-9a-f]{2}(?![0-9a-f])")),
    (
        "private sample fingerprint",
        _PRIVATE_SAMPLE_KEY_PATTERN,
    ),
)
PRIVATE_SAMPLE_BYTE_PATTERN = re.compile(
    b"(?:"
    + b"|".join(prefix.encode("ascii") for prefix in _PRIVATE_SAMPLE_PREFIXES)
    + rb")[^\x00\r\n]{0,96}sha"
    + b"256",
    re.IGNORECASE,
)
GENERIC_LOCAL_ACCOUNT_NAMES = {"admin", "administrator", "defaultuser0", "user"}


class PrivacyGateError(RuntimeError):
    """Raised when a tree contains content forbidden from public release."""


def private_machine_markers() -> tuple[bytes, ...]:
    values: set[str] = {os.path.normpath(str(Path.home()))}
    values.update(
        os.path.normpath(value)
        for name in ("USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP")
        if (value := os.environ.get(name))
    )
    markers: set[bytes] = set()
    for value in values:
        spellings = {value, value.replace("\\", "/"), value.replace("/", "\\")}
        spellings.update(spelling.replace("\\", "\\\\") for spelling in tuple(spellings))
        for spelling in spellings:
            if len(spelling) >= 8:
                markers.add(spelling.encode("utf-8"))
                markers.add(spelling.encode("utf-16-le"))
    computer_name = os.environ.get("COMPUTERNAME", "").strip()
    if len(computer_name) >= 8:
        for spelling in {computer_name, computer_name.casefold(), computer_name.upper()}:
            markers.add(spelling.encode("utf-8"))
            markers.add(spelling.encode("utf-16-le"))
    return tuple(sorted(markers, key=len, reverse=True))


def is_link_or_reparse(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes
    except AttributeError:
        attributes = 0
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _check_private_path(relative: Path, *, allow_arena_engine_config: bool) -> None:
    components = {part.casefold() for part in relative.parts[:-1]}
    forbidden_components = sorted(components & FORBIDDEN_PRIVATE_COMPONENTS)
    if forbidden_components:
        raise PrivacyGateError(
            f"candidate contains a private runtime path component: {relative.as_posix()} ({forbidden_components})"
        )
    if "config" in components and not (
        allow_arena_engine_config and tuple(part.casefold() for part in relative.parts[:2]) == ("assets", "arena-winrate")
    ):
        raise PrivacyGateError(f"candidate contains a non-engine config directory: {relative.as_posix()}")
    name = relative.name.casefold()
    if name in FORBIDDEN_PRIVATE_NAMES or relative.suffix.casefold() in FORBIDDEN_PRIVATE_SUFFIXES:
        raise PrivacyGateError(f"candidate contains a private runtime file: {relative.as_posix()}")


def _scan_bytes(
    relative: Path,
    payload: bytes,
    machine_markers: tuple[bytes, ...],
    *,
    scan_personal_text: bool,
    allowed_emails: Collection[str],
    compressed_image_payload: bool = False,
) -> None:
    if any(marker in payload for marker in machine_markers):
        raise PrivacyGateError(f"candidate contains a local machine path or profile marker: {relative.as_posix()}")
    if any(pattern.search(payload) for pattern in SECRET_BYTE_PATTERNS):
        raise PrivacyGateError(f"candidate contains credential or private-key material: {relative.as_posix()}")
    if PRIVATE_SAMPLE_BYTE_PATTERN.search(payload):
        raise PrivacyGateError(f"candidate contains a private sample fingerprint: {relative.as_posix()}")
    lowered = payload.lower()
    unlicensed_posix = b".local/" + b"unlicensed"
    unlicensed_windows = b".local" + b"\\unlicensed"
    if unlicensed_posix in lowered or unlicensed_windows in lowered:
        raise PrivacyGateError(f"candidate references an unlicensed local source: {relative.as_posix()}")
    if scan_personal_text:
        email_pattern = (
            BINARY_EMAIL_PATTERN
            if compressed_image_payload
            else EMAIL_PATTERN
        )
        text_run_patterns = (
            (ASCII_TEXT_RUN_PATTERN, slice(None)),
            (UTF16_LE_TEXT_RUN_PATTERN, slice(None, None, 2)),
            (UTF16_BE_TEXT_RUN_PATTERN, slice(1, None, 2)),
        )
        for pattern, payload_slice in text_run_patterns:
            for match in pattern.finditer(payload):
                _scan_personal_text(
                    relative,
                    match.group(0)[payload_slice].decode("ascii"),
                    allowed_emails,
                    email_pattern=email_pattern,
                )


def _scan_personal_text(
    relative: Path,
    text: str,
    allowed_emails: Collection[str],
    *,
    email_pattern: re.Pattern[str] = EMAIL_PATTERN,
) -> None:
    allowed = {email.casefold() for email in allowed_emails}
    for match in email_pattern.finditer(text):
        if match.group(0).casefold() not in allowed:
            raise PrivacyGateError(f"project text contains email address: {relative.as_posix()}")
    for label, pattern in PERSONAL_TEXT_PATTERNS:
        if pattern.search(text):
            raise PrivacyGateError(f"project text contains {label}: {relative.as_posix()}")
    username = os.environ.get("USERNAME", "").strip()
    if len(username) >= 4 and username.casefold() not in GENERIC_LOCAL_ACCOUNT_NAMES:
        account_pattern = re.compile(
            rf"(?i)[\"']?(?:user(?:name)?|account(?:_name)?|owner)[\"']?"
            rf"\s*[:=]\s*[\"']?{re.escape(username)}(?![\w.-])"
        )
        if account_pattern.search(text):
            raise PrivacyGateError(
                f"project text contains a local account identifier: {relative.as_posix()}"
            )
    user_domain = os.environ.get("USERDOMAIN", "").strip()
    if len(username) >= 4 and user_domain and re.search(
        rf"(?i)(?<![\w.-]){re.escape(user_domain)}[\\/]{re.escape(username)}(?![\w.-])",
        text,
    ):
        raise PrivacyGateError(
            f"project text contains a local account identifier: {relative.as_posix()}"
        )


def validate_relative_path(
    relative: Path,
    *,
    is_directory: bool,
    allowed_emails: Collection[str] = (),
    allow_arena_engine_config: bool = True,
    machine_markers: tuple[bytes, ...] | None = None,
) -> None:
    """Apply the content privacy rules to one relative path string."""

    checked_path = relative / "_" if is_directory else relative
    _check_private_path(
        checked_path,
        allow_arena_engine_config=allow_arena_engine_config,
    )
    markers = private_machine_markers() if machine_markers is None else machine_markers
    path_text = relative.as_posix()
    _scan_bytes(
        relative,
        path_text.encode("utf-8"),
        markers,
        scan_personal_text=True,
        allowed_emails=allowed_emails,
    )
    _scan_personal_text(relative, path_text, allowed_emails)


def _scan_numpy_text(relative: Path, payload: bytes, allowed_emails: Collection[str]) -> None:
    try:
        import numpy as np

        array = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as error:
        raise PrivacyGateError(f"nested NumPy member cannot be inspected: {relative.as_posix()}") from error
    if array.dtype.kind == "U":
        for value in array.reshape(-1):
            _scan_personal_text(relative, str(value), allowed_emails)
    elif array.dtype.kind == "S":
        for value in array.reshape(-1):
            _scan_personal_text(relative, bytes(value).decode("utf-8", errors="ignore"), allowed_emails)


def _scan_archive(
    path: Path,
    relative: Path,
    machine_markers: tuple[bytes, ...],
    *,
    scan_personal_text: bool,
    allowed_emails: Collection[str],
    allow_embedded_container: bool,
) -> None:
    recognized_suffix = path.suffix.casefold() in ARCHIVE_SUFFIXES
    is_archive = zipfile.is_zipfile(path)
    if is_archive and not recognized_suffix and not allow_embedded_container:
        raise PrivacyGateError(
            f"candidate contains a ZIP container with an unsupported suffix: {relative.as_posix()}"
        )
    if not recognized_suffix and not allow_embedded_container:
        return
    if not is_archive:
        raise PrivacyGateError(f"candidate contains an invalid nested archive: {relative.as_posix()}")
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.comment:
                _scan_bytes(
                    relative / "<archive-comment>",
                    archive.comment,
                    machine_markers,
                    scan_personal_text=scan_personal_text,
                    allowed_emails=allowed_emails,
                )
            names: set[str] = set()
            for entry in archive.infolist():
                member = Path(entry.filename.replace("\\", "/"))
                mode = entry.external_attr >> 16
                if member.is_absolute() or ".." in member.parts or stat.S_ISLNK(mode):
                    raise PrivacyGateError(f"nested archive contains an unsafe member: {relative.as_posix()}!{entry.filename}")
                normalized = member.as_posix()
                if normalized in names:
                    raise PrivacyGateError(f"nested archive contains a duplicate member: {relative.as_posix()}!{normalized}")
                names.add(normalized)
                member_relative = relative / member
                validate_relative_path(
                    member_relative,
                    is_directory=entry.is_dir(),
                    allowed_emails=allowed_emails,
                    allow_arena_engine_config=False,
                    machine_markers=machine_markers,
                )
                for label, metadata in (("comment", entry.comment), ("extra", entry.extra)):
                    if metadata:
                        _scan_bytes(
                            member_relative / f"<{label}>",
                            metadata,
                            machine_markers,
                            scan_personal_text=scan_personal_text,
                            allowed_emails=allowed_emails,
                        )
                if entry.is_dir():
                    continue
                if entry.file_size > 256 * 1024 * 1024:
                    raise PrivacyGateError(
                        f"nested archive entry is too large for the privacy gate: {relative.as_posix()}!{entry.filename}"
                    )
                payload = archive.read(entry)
                if zipfile.is_zipfile(io.BytesIO(payload)):
                    try:
                        with zipfile.ZipFile(io.BytesIO(payload)) as nested_archive:
                            nested_members = nested_archive.infolist()
                    except zipfile.BadZipFile as error:
                        raise PrivacyGateError(
                            f"nested archive contains an invalid archive container: {member_relative.as_posix()}"
                        ) from error
                    if nested_members:
                        raise PrivacyGateError(
                            f"nested archive contains an unsupported archive container: {member_relative.as_posix()}"
                        )
                _scan_bytes(
                    member_relative,
                    payload,
                    machine_markers,
                    scan_personal_text=scan_personal_text,
                    allowed_emails=allowed_emails,
                )
                if scan_personal_text and member.suffix.casefold() in TEXT_SUFFIXES:
                    try:
                        member_text = payload.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise PrivacyGateError(
                            f"nested project text is not valid UTF-8: {member_relative.as_posix()}"
                        ) from error
                    _scan_personal_text(member_relative, member_text, allowed_emails)
                elif scan_personal_text and member.suffix.casefold() == ".npy":
                    _scan_numpy_text(member_relative, payload, allowed_emails)
    except zipfile.BadZipFile as error:
        raise PrivacyGateError(f"candidate contains an invalid nested archive: {relative.as_posix()}") from error


def _validate_image_metadata(path: Path, relative: Path) -> None:
    if path.suffix.casefold() not in IMAGE_SUFFIXES:
        return
    try:
        from PIL import Image

        with Image.open(path) as image:
            extra = {
                key
                for key in image.info
                if key.casefold()
                not in {"background", "dpi", "duration", "gamma", "icc_profile", "loop", "srgb", "transparency"}
            }
            if extra or image.getexif():
                raise PrivacyGateError(
                    f"project image contains nonessential metadata: {relative.as_posix()} keys={sorted(extra)}"
                )
    except PrivacyGateError:
        raise
    except Exception as error:
        raise PrivacyGateError(f"project image metadata cannot be inspected: {relative.as_posix()}") from error


def _validate_embedded_png_metadata(path: Path, relative: Path) -> None:
    if path.suffix.casefold() != ".npz":
        return
    try:
        import numpy as np
        from PIL import Image

        with np.load(path, allow_pickle=False) as arrays:
            if not {"png_bytes", "png_offsets"}.issubset(arrays.files):
                return
            png_bytes = arrays["png_bytes"]
            offsets = arrays["png_offsets"]
            if png_bytes.ndim != 1 or png_bytes.dtype != np.uint8:
                raise PrivacyGateError(f"embedded PNG byte storage is invalid: {relative.as_posix()}")
            if offsets.ndim != 1 or offsets.size < 2 or offsets[0] != 0 or offsets[-1] != png_bytes.size:
                raise PrivacyGateError(f"embedded PNG offsets are invalid: {relative.as_posix()}")
            if np.any(offsets[1:] < offsets[:-1]):
                raise PrivacyGateError(f"embedded PNG offsets are not monotonic: {relative.as_posix()}")
            for index in range(offsets.size - 1):
                start = int(offsets[index])
                end = int(offsets[index + 1])
                with Image.open(io.BytesIO(png_bytes[start:end].tobytes())) as image:
                    extra = {
                        key
                        for key in image.info
                        if key.casefold()
                        not in {"background", "dpi", "duration", "gamma", "icc_profile", "loop", "srgb", "transparency"}
                    }
                    if extra or image.getexif():
                        raise PrivacyGateError(
                            f"embedded project image contains nonessential metadata: "
                            f"{relative.as_posix()}#png[{index}] keys={sorted(extra)}"
                        )
    except PrivacyGateError:
        raise
    except Exception as error:
        raise PrivacyGateError(f"embedded PNG metadata cannot be inspected: {relative.as_posix()}") from error


def validate_tree(
    root: Path,
    *,
    project_path_predicate: Callable[[Path], bool],
    allowed_emails: Collection[str] = (),
    allow_arena_engine_config: bool = True,
    allowed_embedded_archives: Collection[str] = (),
) -> dict[str, int]:
    """Validate the exact files present below ``root`` and return scan counts."""

    markers = private_machine_markers()
    embedded_archives = {path.casefold() for path in allowed_embedded_archives}
    files = 0
    bytes_scanned = 0
    for entry in root.rglob("*"):
        relative = entry.relative_to(root)
        if is_link_or_reparse(entry):
            raise PrivacyGateError(f"candidate contains a link or reparse point: {relative.as_posix()}")
        validate_relative_path(
            relative,
            is_directory=entry.is_dir(),
            allowed_emails=allowed_emails,
            allow_arena_engine_config=allow_arena_engine_config,
            machine_markers=markers,
        )
    for path in sorted(entry for entry in root.rglob("*") if entry.is_file()):
        relative = path.relative_to(root)
        payload = path.read_bytes()
        files += 1
        bytes_scanned += len(payload)
        is_project_path = project_path_predicate(relative)
        is_project_image = is_project_path and path.suffix.casefold() in IMAGE_SUFFIXES
        if is_project_image:
            _validate_image_metadata(path, relative)
        _scan_bytes(
            relative,
            payload,
            markers,
            scan_personal_text=is_project_path,
            allowed_emails=allowed_emails,
            compressed_image_payload=is_project_image,
        )
        _scan_archive(
            path,
            relative,
            markers,
            scan_personal_text=is_project_path,
            allowed_emails=allowed_emails,
            allow_embedded_container=relative.as_posix().casefold() in embedded_archives,
        )
        if is_project_path and path.suffix.casefold() in TEXT_SUFFIXES:
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise PrivacyGateError(f"project text is not valid UTF-8: {relative.as_posix()}") from error
            _scan_personal_text(relative, text, allowed_emails)
        if is_project_path:
            _validate_embedded_png_metadata(path, relative)
    return {"files_scanned": files, "bytes_scanned": bytes_scanned}
