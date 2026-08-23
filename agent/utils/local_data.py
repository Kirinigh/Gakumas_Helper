"""Bounded, local-only capture storage and structured decision events.

The capture writer is deliberately opt-in. It accepts bytes supplied by a
caller but never acquires a screen image itself, and it has no network path.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

MIB = 1024 * 1024
GIB = 1024 * MIB
CAPTURE_SCHEMA_VERSION = "1.0"
DECISION_SCHEMA_VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / ".local" / "runtime-data"
CAPTURE_MODES = ("off", "failure", "roi", "screenshot")

_SECRET_FRAGMENTS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "cookie",
    "authorization",
    "credential",
    "api_key",
    "apikey",
    "account",
)
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp", ".bin"}
_CAPTURE_KINDS = {"screenshot", "roi", "failure_scene"}
_MODE_CAPTURE_KINDS = {
    "off": set(),
    "failure": {"failure_scene"},
    "roi": {"roi", "failure_scene"},
    "screenshot": {"screenshot", "roi", "failure_scene"},
}
_ATOMIC_TEMPORARY_NAME = re.compile(r".+\.tmp-[0-9a-f]{32}", re.DOTALL)
_ROOT_LOCKS: dict[str, threading.RLock] = {}
_ROOT_LOCKS_GUARD = threading.Lock()
_RUNTIME_LOCK = threading.RLock()
_RUNTIME_CONFIG = None
_RUNTIME_STORE = None


class LocalDataError(RuntimeError):
    """Base class for bounded local-data failures."""


class CaptureDisabledError(LocalDataError):
    """Raised when a caller attempts capture without explicit opt-in."""


class StorageSafetyError(LocalDataError):
    """Raised when bounds cannot be proved before or after a write."""


class SecretFieldError(ValueError):
    """Raised when structured data contains a forbidden secret-like field."""


@dataclass(frozen=True)
class RetentionPolicy:
    max_age_days: int
    max_bytes: int

    def __post_init__(self) -> None:
        if self.max_age_days < 0 or self.max_bytes < 0:
            raise ValueError("retention limits must be non-negative")


@dataclass(frozen=True)
class LocalDataConfig:
    enabled: bool = False
    mode: str = "off"
    root: Path = DEFAULT_RUNTIME_ROOT
    normal: RetentionPolicy = field(default_factory=lambda: RetentionPolicy(14, 512 * MIB))
    failure: RetentionPolicy = field(default_factory=lambda: RetentionPolicy(30, 512 * MIB))
    total_image_bytes: int = GIB
    total_metadata_bytes: int = 64 * MIB
    total_artifact_bytes: int = GIB + 64 * MIB
    minimum_free_bytes: int = 16 * MIB
    log_retention_days: int = 14

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        root_text = str(self.root)
        if root_text.startswith(("\\", "//")):
            raise ValueError("capture root must not be a network path")
        if self.mode not in CAPTURE_MODES:
            raise ValueError(f"capture mode must be one of {CAPTURE_MODES}")
        if self.enabled != (self.mode != "off"):
            raise ValueError("enabled and capture mode must agree")
        if (
            self.total_image_bytes < 0
            or self.total_metadata_bytes < 0
            or self.total_artifact_bytes < 0
            or self.minimum_free_bytes < 0
            or self.log_retention_days < 0
        ):
            raise ValueError("storage limits must be non-negative")


@dataclass(frozen=True)
class StorageUsage:
    normal_files: int
    normal_bytes: int
    failure_files: int
    failure_bytes: int
    log_files: int
    log_bytes: int
    metadata_files: int = 0
    metadata_bytes: int = 0

    @property
    def image_files(self) -> int:
        return self.normal_files + self.failure_files

    @property
    def image_bytes(self) -> int:
        return self.normal_bytes + self.failure_bytes

    @property
    def total_files(self) -> int:
        return self.image_files + self.log_files + self.metadata_files

    @property
    def total_bytes(self) -> int:
        return self.image_bytes + self.log_bytes + self.metadata_bytes


@dataclass(frozen=True)
class CleanupReport:
    before: StorageUsage
    after: StorageUsage

    @property
    def deleted_files(self) -> int:
        return self.before.total_files - self.after.total_files

    @property
    def deleted_bytes(self) -> int:
        return self.before.total_bytes - self.after.total_bytes


@dataclass(frozen=True)
class CaptureSession:
    session_id: str
    started_at: str
    frame_version: str
    schema_version: str = CAPTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.session_id) != 32 or any(character not in "0123456789abcdef" for character in self.session_id):
            raise ValueError("session_id must be a 32-character lowercase hexadecimal identifier")
        if not self.frame_version.strip():
            raise ValueError("frame_version is required")
        if self.schema_version != CAPTURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported capture schema version: {self.schema_version}")
        _parse_utc(self.started_at)

    @classmethod
    def create(cls, frame_version: str, now: datetime | None = None) -> CaptureSession:
        frame_version = frame_version.strip()
        if not frame_version:
            raise ValueError("frame_version is required")
        instant = _utc(now)
        return cls(uuid.uuid4().hex, instant.isoformat(), frame_version)


@dataclass(frozen=True)
class DecisionEvent:
    domain: str
    event_id: str
    occurred_at: str
    input_version: str
    candidates: tuple[Mapping[str, Any], ...]
    final_action: Mapping[str, Any] | None
    filters: tuple[Mapping[str, Any], ...] = ()
    fallback: Mapping[str, Any] | None = None
    context: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.domain not in {"card", "live", "route", "arena"}:
            raise ValueError("domain must be card, live, route, or arena")
        if not self.event_id or not self.input_version:
            raise ValueError("event_id and input_version are required")
        _parse_utc(self.occurred_at)
        reject_secret_fields(asdict(self))

    def to_json(self) -> str:
        payload = asdict(self)
        reject_secret_fields(payload)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def reject_secret_fields(value: Any, path: str = "$") -> None:
    """Recursively reject field names that may carry credentials or identity secrets."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"structured field name must be text at {path}")
            normalized = key.strip().lower().replace("-", "_")
            if any(fragment in normalized for fragment in _SECRET_FRAGMENTS):
                raise SecretFieldError(f"forbidden secret field at {path}.{key}")
            reject_secret_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_secret_fields(child, f"{path}[{index}]")


class LocalCaptureStore:
    """Thread-safe, quota-bound writer for caller-supplied synthetic or real bytes."""

    _CATEGORIES = ("normal", "failure")

    def __init__(
        self,
        config: LocalDataConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        disk_usage: Callable[[Path], Any] | None = None,
    ) -> None:
        self.config = config
        self._clock = clock or (lambda: datetime.now(UTC))
        self._disk_usage = disk_usage or shutil.disk_usage
        self._lock = _shared_root_lock(config.root)
        self._blocked_reason: str | None = None
        self._revoked = False

    @property
    def blocked_reason(self) -> str | None:
        return self._blocked_reason

    @property
    def revoked(self) -> bool:
        return self._revoked

    def revoke(self) -> None:
        """Permanently invalidate cached references after runtime reconfiguration."""

        with self._lock:
            self._revoked = True

    def usage(self) -> StorageUsage:
        """Return current image, metadata, and log usage for status notifications."""

        if not self.config.root.exists():
            return StorageUsage(0, 0, 0, 0, 0, 0, 0, 0)
        with self._storage_guard():
            return self._usage_locked()

    def create_evidence_file(self, target: Path, data: bytes) -> Path:
        """Create one immutable bounded annotation or frozen-manifest file."""

        self._require_enabled()
        if not isinstance(data, bytes):
            raise TypeError("evidence data must be bytes")
        try:
            structured = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("evidence file must contain UTF-8 JSON") from error
        reject_secret_fields(structured)
        target = Path(target).resolve()
        root = self.config.root.resolve()
        try:
            relative = target.relative_to(root)
        except ValueError as error:
            raise ValueError("evidence file must stay under the configured runtime root") from error
        parts = relative.parts
        frozen = len(parts) == 2 and parts[0] == "frozen" and target.suffix.lower() == ".json"
        annotation = (
            len(parts) == 4
            and parts[0] == "images"
            and parts[1] in self._CATEGORIES
            and len(parts[2]) == 32
            and all(character in "0123456789abcdef" for character in parts[2])
            and target.name.endswith(".png.json.annotation.json")
        )
        if not (frozen or annotation):
            raise ValueError("unsupported evidence file path")
        try:
            with self._storage_guard():
                self._require_writable_locked()
                if target.exists():
                    raise FileExistsError(f"refusing to overwrite immutable evidence: {target}")
                if self._metadata_bytes() + len(data) > self.config.total_metadata_bytes:
                    raise StorageSafetyError("immutable evidence exceeds the metadata hard limit")
                if self._artifact_bytes() + len(data) > self.config.total_artifact_bytes:
                    raise StorageSafetyError("immutable evidence exceeds the total artifact hard limit")
                self._ensure_space(len(data))
                self._atomic_create(target, data)
                if (
                    not target.exists()
                    or self._metadata_bytes() > self.config.total_metadata_bytes
                    or self._artifact_bytes() > self.config.total_artifact_bytes
                ):
                    raise StorageSafetyError("immutable evidence could not be retained within hard limits")
            return target
        except FileExistsError:
            raise
        except Exception as error:
            try:
                target.unlink(missing_ok=True)
            except OSError as rollback_error:
                error = StorageSafetyError(f"{error}; rollback failed: {rollback_error}")
            self._suspend(error)
            raise StorageSafetyError(self._blocked_reason or "evidence storage suspended") from error

    def start_session(self, frame_version: str) -> CaptureSession:
        self._require_enabled()
        session = CaptureSession.create(frame_version, self._clock())
        target = self.config.root / "sessions" / f"{session.session_id}.json"
        session_bytes = json.dumps(asdict(session), ensure_ascii=False, indent=2).encode("utf-8")
        try:
            with self._storage_guard():
                self._require_writable_locked()
                self._cleanup_locked(incoming_bytes=len(session_bytes))
                self._ensure_space(len(session_bytes))
                self._atomic_write(target, session_bytes)
                self._cleanup_locked()
                if (
                    not target.exists()
                    or self._metadata_bytes() > self.config.total_metadata_bytes
                    or self._artifact_bytes() > self.config.total_artifact_bytes
                ):
                    raise StorageSafetyError("session metadata could not be retained within hard limits")
            return session
        except Exception as error:
            detail: Exception | str = error
            try:
                target.unlink(missing_ok=True)
            except OSError as rollback_error:
                detail = f"{error}; rollback failed: {rollback_error}"
            self._suspend(detail)
            raise StorageSafetyError(self._blocked_reason or "capture suspended") from error

    def write_image(
        self,
        category: str,
        session: CaptureSession,
        data: bytes,
        *,
        metadata: Mapping[str, Any] | None = None,
        capture_kind: str | None = None,
        suffix: str = ".bin",
    ) -> Path:
        self._require_enabled()
        if not isinstance(session, CaptureSession):
            raise TypeError("session must be a validated CaptureSession")
        if category not in self._CATEGORIES:
            raise ValueError(f"unknown capture category: {category}")
        if not isinstance(data, bytes):
            raise TypeError("capture data must be bytes")
        suffix = suffix.lower()
        if suffix not in _IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image suffix: {suffix}")
        if capture_kind is None:
            raise ValueError("capture_kind must be explicit")
        if capture_kind not in _CAPTURE_KINDS:
            raise ValueError(f"unknown capture kind: {capture_kind}")
        if (category == "failure") != (capture_kind == "failure_scene"):
            raise ValueError("failure_scene must use failure retention; other kinds must use normal retention")
        if capture_kind not in _MODE_CAPTURE_KINDS[self.config.mode]:
            raise CaptureDisabledError(f"capture kind {capture_kind} is disabled by mode {self.config.mode}")
        metadata = dict(metadata or {})
        reject_secret_fields(metadata)

        target: Path | None = None
        manifest: Path | None = None
        try:
            stamp = _utc(self._clock())
            stem = f"{stamp.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}"
            target = self.config.root / "images" / category / session.session_id / f"{stem}{suffix}"
            manifest = target.with_suffix(target.suffix + ".json")
            manifest_payload = {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "session_id": session.session_id,
                "captured_at": stamp.isoformat(),
                "frame_version": session.frame_version,
                "category": category,
                "capture_kind": capture_kind,
                "image_file": target.name,
                "byte_size": len(data),
                "metadata": metadata,
            }
            manifest_bytes = json.dumps(manifest_payload, ensure_ascii=False, indent=2).encode("utf-8")
            incoming_total = len(data) + len(manifest_bytes)
            with self._storage_guard():
                self._require_writable_locked()
                self._validate_session_sidecar(session)
                self._cleanup_locked(
                    incoming_bytes=incoming_total,
                    incoming_category=category,
                    incoming_image_bytes=len(data),
                    protected_session_id=session.session_id,
                )
                self._ensure_space(incoming_total)
                self._atomic_write(target, data)
                self._atomic_write(manifest, manifest_bytes)
                self._cleanup_locked(protected_session_id=session.session_id)
                if (
                    not target.exists()
                    or not manifest.exists()
                    or self._metadata_bytes() > self.config.total_metadata_bytes
                    or self._artifact_bytes() > self.config.total_artifact_bytes
                ):
                    raise StorageSafetyError("new capture could not be retained within hard limits")
                return target
        except Exception as error:
            rollback_errors: list[str] = []
            for path in (target, manifest):
                if path is None:
                    continue
                try:
                    path.unlink(missing_ok=True)
                except OSError as rollback_error:
                    rollback_errors.append(str(rollback_error))
            detail: Exception | str = error
            if rollback_errors:
                detail = f"{error}; rollback failed: {'; '.join(rollback_errors)}"
            self._suspend(detail)
            raise StorageSafetyError(self._blocked_reason or "capture suspended") from error

    def cleanup(
        self,
        *,
        incoming_bytes: int = 0,
        incoming_category: str | None = None,
        incoming_image_bytes: int | None = None,
    ) -> CleanupReport:
        """Enforce age, category image caps, and the total artifact cap oldest-first."""

        if incoming_bytes < 0:
            raise ValueError("incoming_bytes must be non-negative")
        if incoming_image_bytes is None:
            incoming_image_bytes = incoming_bytes if incoming_category is not None else 0
        if incoming_image_bytes < 0 or incoming_image_bytes > incoming_bytes:
            raise ValueError("incoming_image_bytes must be within incoming_bytes")
        if incoming_category is not None and incoming_category not in self._CATEGORIES:
            raise ValueError(f"unknown capture category: {incoming_category}")
        if not self.config.root.exists():
            empty = StorageUsage(0, 0, 0, 0, 0, 0, 0, 0)
            return CleanupReport(empty, empty)
        with self._storage_guard():
            before = self._usage_locked()
            self._cleanup_locked(
                incoming_bytes=incoming_bytes,
                incoming_category=incoming_category,
                incoming_image_bytes=incoming_image_bytes,
            )
            return CleanupReport(before, self._usage_locked())

    def _cleanup_locked(
        self,
        *,
        incoming_bytes: int = 0,
        incoming_category: str | None = None,
        incoming_image_bytes: int | None = None,
        protected_session_id: str | None = None,
    ) -> None:
        if incoming_image_bytes is None:
            incoming_image_bytes = incoming_bytes if incoming_category is not None else 0
        incoming_metadata_bytes = incoming_bytes - incoming_image_bytes
        if incoming_bytes > self.config.total_artifact_bytes:
            raise OSError("incoming artifact exceeds the total hard limit")
        if incoming_metadata_bytes > self.config.total_metadata_bytes:
            raise OSError("incoming metadata exceeds the metadata hard limit")
        if incoming_category is not None:
            policy = getattr(self.config, incoming_category)
            if incoming_image_bytes > policy.max_bytes:
                raise OSError("incoming capture exceeds a category image limit")

        now = _utc(self._clock())
        for category in self._CATEGORIES:
            policy = getattr(self.config, category)
            files = self._image_files(category)
            cutoff = (now - timedelta(days=policy.max_age_days)).timestamp()
            for path in files:
                if path.stat().st_mtime < cutoff:
                    self._remove_capture(path)
            files = self._image_files(category)
            allowed = policy.max_bytes - (incoming_image_bytes if category == incoming_category else 0)
            self._trim_to_image_limit(files, allowed)

        log_cutoff = (now - timedelta(days=self.config.log_retention_days)).timestamp()
        for path in _files_under(self.config.root / "logs"):
            if path.stat().st_mtime < log_cutoff:
                path.unlink(missing_ok=True)

        metadata_cutoff = (now - timedelta(days=max(self.config.normal.max_age_days, self.config.failure.max_age_days))).timestamp()
        active_sessions = {path.parent.name for path in self._image_files()}
        for path in self._session_files():
            if path.stem == protected_session_id or path.stem in active_sessions:
                continue
            if path.stat().st_mtime < metadata_cutoff:
                path.unlink(missing_ok=True)
        self._remove_orphan_manifests()
        self._trim_metadata_to_limit(
            self.config.total_metadata_bytes - incoming_metadata_bytes,
            protected_session_id=protected_session_id,
        )

        total_image_allowed = self.config.total_image_bytes - incoming_image_bytes
        self._trim_to_image_limit(self._image_files(), total_image_allowed)
        allowed_total = self.config.total_artifact_bytes - incoming_bytes
        self._trim_artifacts_to_limit(allowed_total, protected_session_id=protected_session_id)

    def reset_after_operator_check(self) -> None:
        """Resume writes only after a caller has explicitly handled the prior storage fault."""

        with self._storage_guard():
            if self._revoked:
                raise CaptureDisabledError("local capture store was revoked by reconfiguration")
            self._cleanup_locked()
            self._blocked_reason = None

    def _require_enabled(self) -> None:
        with self._lock:
            if self._revoked:
                raise CaptureDisabledError("local capture store was revoked by reconfiguration")
            if not self.config.enabled:
                raise CaptureDisabledError("local capture is disabled; explicit opt-in is required")

    def _require_writable_locked(self) -> None:
        if self._revoked:
            raise CaptureDisabledError("local capture store was revoked by reconfiguration")
        if self._blocked_reason is not None:
            raise StorageSafetyError(self._blocked_reason)

    def _suspend(self, error: Exception | str) -> None:
        if isinstance(error, StorageSafetyError) and self._blocked_reason is not None:
            return
        with self._lock:
            self._blocked_reason = f"capture suspended after storage failure: {error}"

    def _prepare_root(self) -> None:
        self.config.root.mkdir(parents=True, exist_ok=True)

    def _validate_session_sidecar(self, session: CaptureSession) -> None:
        target = self.config.root / "sessions" / f"{session.session_id}.json"
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageSafetyError("capture session is not registered in this store") from error
        if payload != asdict(session):
            raise StorageSafetyError("capture session sidecar does not match the supplied session")

    @contextmanager
    def _storage_guard(self) -> Iterator[None]:
        with self._lock:
            self._prepare_root()
            with _interprocess_lock(self.config.root / ".quota.lock"):
                self._remove_stale_temporaries_locked()
                yield

    def _usage_locked(self) -> StorageUsage:
        normal_files, normal_bytes = _path_usage(self._image_files("normal"))
        failure_files, failure_bytes = _path_usage(self._image_files("failure"))
        log_files, log_bytes = _path_usage(_files_under(self.config.root / "logs"))
        metadata_files, metadata_bytes = _path_usage(self._metadata_files())
        return StorageUsage(
            normal_files,
            normal_bytes,
            failure_files,
            failure_bytes,
            log_files,
            log_bytes,
            metadata_files,
            metadata_bytes,
        )

    def _remove_stale_temporaries_locked(self) -> None:
        """Remove only abandoned files produced by this store's atomic writers."""

        for name in ("images", "sessions", "frozen"):
            root = self.config.root / name
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and _ATOMIC_TEMPORARY_NAME.fullmatch(path.name):
                    path.unlink(missing_ok=True)

    def _ensure_space(self, incoming_bytes: int) -> None:
        free = int(self._disk_usage(self.config.root).free)
        if free - incoming_bytes < self.config.minimum_free_bytes:
            raise OSError("insufficient free disk space for a bounded capture")

    def _image_files(self, category: str | None = None) -> list[Path]:
        base = self.config.root / "images"
        roots = [base / category] if category else [base / item for item in self._CATEGORIES]
        files: list[Path] = []
        for root in roots:
            if root.exists():
                files.extend(
                    path
                    for path in root.rglob("*")
                    if path.is_file() and not path.name.endswith(".json") and ".tmp-" not in path.name
                )
        return sorted(files, key=lambda path: (path.stat().st_mtime, str(path)))

    def _manifest_files(self) -> list[Path]:
        root = self.config.root / "images"
        if not root.exists():
            return []
        return sorted(
            (
                path
                for path in root.rglob("*.json")
                if path.is_file()
                and not path.name.endswith(".annotation.json")
                and ".tmp-" not in path.name
            ),
            key=lambda path: (path.stat().st_mtime, str(path)),
        )

    def _session_files(self) -> list[Path]:
        return sorted(
            (
                path
                for path in _files_under(self.config.root / "sessions")
                if path.suffix.lower() == ".json" and ".tmp-" not in path.name
            ),
            key=lambda path: (path.stat().st_mtime, str(path)),
        )

    def _metadata_files(self) -> list[Path]:
        return self._session_files() + self._manifest_files() + self._annotation_files() + self._frozen_files()

    def _annotation_files(self) -> list[Path]:
        root = self.config.root / "images"
        if not root.exists():
            return []
        return sorted(
            (
                path
                for path in root.rglob("*.annotation.json")
                if path.is_file() and ".tmp-" not in path.name
            ),
            key=lambda path: (path.stat().st_mtime, str(path)),
        )

    def _frozen_files(self) -> list[Path]:
        return sorted(
            (
                path
                for path in _files_under(self.config.root / "frozen")
                if path.suffix.lower() == ".json" and ".tmp-" not in path.name
            ),
            key=lambda path: (path.stat().st_mtime, str(path)),
        )

    def _artifact_files(self) -> list[Path]:
        return self._image_files() + self._metadata_files()

    def _artifact_bytes(self) -> int:
        return sum(path.stat().st_size for path in self._artifact_files())

    def _metadata_bytes(self) -> int:
        return sum(path.stat().st_size for path in self._metadata_files())

    def _trim_to_image_limit(self, files: list[Path], limit: int) -> None:
        if limit < 0:
            for path in files:
                self._remove_capture(path)
            return
        total = sum(path.stat().st_size for path in files)
        for path in files:
            if total <= limit:
                break
            size = path.stat().st_size
            self._remove_capture(path)
            total -= size

    def _trim_artifacts_to_limit(self, limit: int, *, protected_session_id: str | None = None) -> None:
        if limit < 0:
            limit = 0
        while self._artifact_bytes() > limit:
            captures = self._image_files()
            if not captures:
                raise OSError("metadata cannot fit within the total artifact limit")
            self._remove_capture(captures[0])
            self._remove_orphan_manifests()
            self._remove_unreferenced_sessions(protected_session_id=protected_session_id)

    def _trim_metadata_to_limit(self, limit: int, *, protected_session_id: str | None = None) -> None:
        if limit < 0:
            limit = 0
        while self._metadata_bytes() > limit:
            self._remove_unreferenced_sessions(protected_session_id=protected_session_id)
            if self._metadata_bytes() <= limit:
                return
            captures = self._image_files()
            if not captures:
                raise OSError("immutable metadata cannot fit within the metadata hard limit")
            self._remove_capture(captures[0])
            self._remove_orphan_manifests()

    def _remove_unreferenced_sessions(self, *, protected_session_id: str | None = None) -> None:
        active_sessions = {path.parent.name for path in self._image_files()}
        for session in self._session_files():
            if session.stem != protected_session_id and session.stem not in active_sessions:
                session.unlink(missing_ok=True)

    def _remove_orphan_manifests(self) -> None:
        for manifest in self._manifest_files():
            image = manifest.with_suffix("")
            if not image.exists():
                manifest.unlink(missing_ok=True)
                manifest.with_suffix(manifest.suffix + ".annotation.json").unlink(missing_ok=True)
        for annotation in self._annotation_files():
            manifest = annotation.with_name(annotation.name.removesuffix(".annotation.json"))
            if not manifest.exists() or not manifest.with_suffix("").exists():
                annotation.unlink(missing_ok=True)

    @staticmethod
    def _remove_capture(path: Path) -> None:
        manifest = path.with_suffix(path.suffix + ".json")
        path.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        manifest.with_suffix(manifest.suffix + ".annotation.json").unlink(missing_ok=True)

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_create(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _utc(parsed)


def _files_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file()]


def _path_usage(paths: list[Path]) -> tuple[int, int]:
    return len(paths), sum(path.stat().st_size for path in paths)


def format_storage_status(config: LocalDataConfig, report: CleanupReport) -> str:
    """Build a privacy-safe status line for Maa logs and notifications."""

    after = report.after
    return (
        f"本地采集模式={config.mode}；路径={config.root}；"
        f"普通图像={after.normal_files} 个/{_format_bytes(after.normal_bytes)}；"
        f"失败现场={after.failure_files} 个/{_format_bytes(after.failure_bytes)}；"
        f"元数据={after.metadata_files} 个/{_format_bytes(after.metadata_bytes)}；"
        f"日志={after.log_files} 个/{_format_bytes(after.log_bytes)}；"
        f"本次清理={report.deleted_files} 个/{_format_bytes(report.deleted_bytes)}"
    )


def config_from_action_params(params: Mapping[str, Any]) -> LocalDataConfig:
    """Translate versioned Maa custom-action parameters into a strict config."""

    if not isinstance(params, Mapping):
        raise TypeError("custom-action parameters must be an object")
    reject_secret_fields(params)
    mode = params.get("capture_mode", "off")
    root = params.get("root", str(DEFAULT_RUNTIME_ROOT))
    if not isinstance(mode, str) or not isinstance(root, str):
        raise ValueError("capture_mode and root must be text")
    requested_root = Path(root)
    root_path = DEFAULT_RUNTIME_ROOT if requested_root == Path(".local/runtime-data") else requested_root.resolve()
    if root_path.resolve() != DEFAULT_RUNTIME_ROOT.resolve():
        raise ValueError(f"capture root must be the ignored local path {DEFAULT_RUNTIME_ROOT}")
    return LocalDataConfig(enabled=mode != "off", mode=mode, root=root_path)


def configure_runtime(params: Mapping[str, Any]) -> tuple[LocalCaptureStore, CleanupReport, str]:
    """Atomically replace the unique store and revoke every cached old reference."""

    try:
        config = config_from_action_params(params)
        candidate = LocalCaptureStore(config)
        report = candidate.cleanup()
    except Exception:
        _replace_runtime(LocalDataConfig())
        raise
    _replace_runtime(config, candidate)
    return candidate, report, format_storage_status(config, report)


def _replace_runtime(config: LocalDataConfig, store: LocalCaptureStore | None = None) -> LocalCaptureStore:
    """Replace runtime references under one lock and permanently revoke the prior store."""

    global _RUNTIME_CONFIG, _RUNTIME_STORE

    store = store or LocalCaptureStore(config)
    with _RUNTIME_LOCK:
        previous = _RUNTIME_STORE
        _RUNTIME_CONFIG = config
        _RUNTIME_STORE = store
        if previous is not None and previous is not store:
            previous.revoke()
    return store


def get_runtime_config() -> LocalDataConfig:
    """Return the active UI-derived config; defaults to an explicit off config."""

    global _RUNTIME_CONFIG

    with _RUNTIME_LOCK:
        if _RUNTIME_CONFIG is None:
            _RUNTIME_CONFIG = LocalDataConfig()
        return _RUNTIME_CONFIG


def get_runtime_store() -> LocalCaptureStore:
    """Return the unique process-wide store for downstream feature tasks."""

    global _RUNTIME_STORE

    with _RUNTIME_LOCK:
        if _RUNTIME_STORE is None:
            _RUNTIME_STORE = LocalCaptureStore(get_runtime_config())
        return _RUNTIME_STORE


def _format_bytes(value: int) -> str:
    return f"{value / MIB:.2f} MiB"


def _shared_root_lock(root: Path) -> threading.RLock:
    key = os.path.normcase(str(root.resolve()))
    with _ROOT_LOCKS_GUARD:
        return _ROOT_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _interprocess_lock(path: Path) -> Iterator[None]:
    """Serialize quota checks across processes using only the standard library."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
