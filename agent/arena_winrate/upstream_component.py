"""Lazy RIS production engine/data component discovery and activation."""

from __future__ import annotations

import os
import re
import json
import time
import uuid
import shutil
import hashlib
import tempfile
import threading
import traceback
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol
from pathlib import Path, PurePosixPath
from dataclasses import dataclass
from collections.abc import Mapping, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import PROJECT_ROOT, DEFAULT_BUNDLE_DIR
from .stages import StageCatalogError, ContestStageCatalog, ContestSeasonDefinition
from .adapter import (
    PROTOCOL_VERSION,
    OWN_SCORE_AGGREGATION,
    AdapterError,
    SubprocessArenaAdapter,
)
from .catalog import ArenaCatalogError, ArenaEntityCatalog
from .season_ui import ArenaSeasonUiSynchronizer
from .badge_reference import BadgeReferenceError, BadgeReferenceGallery
from .component_builder import build_runtime_component

REPOSITORY = "surisuririsu/gakumas-tools"
API_BASE = f"https://api.github.com/repos/{REPOSITORY}"
RAW_BASE = f"https://raw.githubusercontent.com/{REPOSITORY}"
DEFAULT_COMPONENT_ROOT = PROJECT_ROOT / ".local" / "runtime-data" / "arena-components"
HOST_P_ITEM_REFERENCE_RELATIVE_ROOT = (
    Path("resource")
    / "base"
    / "model"
    / "embedding"
    / "p_item_reference"
)
HOST_CARD_REFERENCE_RELATIVE_ROOT = (
    Path("resource")
    / "base"
    / "model"
    / "embedding"
    / "arena_badge_reference"
)
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
STATE_SCHEMA_VERSION = 1
COMPONENT_UPDATE_SCHEMA_VERSION = 1
MAX_API_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_SOURCE_FILE_BYTES = 32 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_SOURCE_FILES = 2000
MAX_PRODUCTION_DEPLOYMENTS = 100
HTTP_ATTEMPTS = 3
FAILED_CHECK_RETRIES = 1
DIRECTORY_MOVE_RETRY_DELAYS = (0.1, 0.3, 0.6)
SOURCE_PREFIXES = (
    "packages/gakumas-engine/",
    "packages/gakumas-data/",
)
REQUIRED_SOURCE_PATHS = {
    "LICENSE",
    "packages/gakumas-engine/package.json",
    "packages/gakumas-data/package.json",
    "packages/gakumas-data/json/stages.json",
}


class ArenaComponentError(RuntimeError):
    """Raised when no installed component can satisfy the selected arena season."""


class ArenaPItemCoverageError(ArenaComponentError):
    """Raised when engine P-item reachability and the host gallery disagree."""


def _log_component_failure(stage: str, error: BaseException, **details: object) -> None:
    """Keep the original chain in the existing detailed log, away from the GUI."""

    try:
        from utils import logger

        context = json.dumps(details, ensure_ascii=False, default=str)
        chain = "".join(traceback.format_exception(error))
        logger.bind(ui_visible=False).error(f"RIS_COMPONENT_FAILURE stage={stage} details={context}\n{chain}")
    except Exception as logging_error:
        error.add_note(f"Component diagnostic logging failed: {logging_error!r}")


def _recoverable_component_failure(error: Exception) -> bool:
    """Retry I/O failures, never unsupported or contradictory catalog semantics."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "_arena_component_retry_exhausted", False):
            return False
        if isinstance(current, ArenaPItemCoverageError):
            return False
        if isinstance(current, urllib.error.HTTPError):
            return current.code >= 500 or current.code in {408, 429}
        if isinstance(current, (OSError, subprocess.TimeoutExpired)):
            return True
        current = current.__cause__
    return False


def _promote_component_directory(source: Path, destination: Path) -> None:
    """Retry only a failed Windows move, without rebuilding a validated bundle."""

    for attempt in range(len(DIRECTORY_MOVE_RETRY_DELAYS) + 1):
        if destination.exists():
            raise FileExistsError(f"component destination already exists: {destination}")
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if destination.exists():
                raise FileExistsError(f"component destination already exists: {destination}") from error
            if (
                getattr(error, "winerror", None) not in {5, 32, 33}
                or attempt == len(DIRECTORY_MOVE_RETRY_DELAYS)
            ):
                error._arena_component_retry_exhausted = True
                error.add_note(f"Directory move stopped after {attempt + 1} attempt(s); do not rebuild to renew this budget")
                raise
            delay = DIRECTORY_MOVE_RETRY_DELAYS[attempt]
            _log_component_failure(
                "promote_retry", error, source=source, destination=destination,
                failed_attempt=attempt + 1, delay_seconds=delay,
            )
            time.sleep(delay)


def _resolve_host_p_item_reference_root() -> Path:
    """Resolve the fixed gallery in source-tree or installed Maa layout."""

    candidates = (
        PROJECT_ROOT / "assets" / HOST_P_ITEM_REFERENCE_RELATIVE_ROOT,
        PROJECT_ROOT / HOST_P_ITEM_REFERENCE_RELATIVE_ROOT,
    )
    return next(
        (
            candidate
            for candidate in candidates
            if (candidate / "manifest.json").is_file()
        ),
        candidates[0],
    )


def _resolve_host_card_reference_root() -> Path:
    """Resolve the fixed card gallery in source-tree or installed Maa layout."""

    candidates = (
        PROJECT_ROOT / "assets" / HOST_CARD_REFERENCE_RELATIVE_ROOT,
        PROJECT_ROOT / HOST_CARD_REFERENCE_RELATIVE_ROOT,
    )
    return next(
        (
            candidate
            for candidate in candidates
            if (candidate / "manifest.json").is_file()
        ),
        candidates[0],
    )


class ArenaUpstreamSource(Protocol):
    def discover_production_commit(self) -> str: ...

    def fetch_stage_rows(self, commit: str) -> object: ...

    def materialize_source(self, commit: str, destination: Path) -> None: ...


@dataclass(frozen=True)
class ArenaComponentResolution:
    bundle_dir: Path
    season: ContestSeasonDefinition
    commit: str
    remote_commit: str | None
    remote_latest_season: int | None
    status: str
    update_activated: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BundleInfo:
    path: Path
    commit: str
    catalog: ContestStageCatalog
    latest_season: int
    host_revision: str | None = None
    reference_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CheckOutcome:
    bundle: _BundleInfo
    remote_commit: str | None
    remote_latest_season: int | None
    status: str
    update_activated: bool
    warnings: tuple[str, ...]
    p_item_coverage_update_failed: bool = False
    failure: Exception | None = None


class GitHubProductionSource:
    """Read immutable RIS files selected by its latest successful deployment."""

    def __init__(
        self,
        *,
        check_timeout_seconds: float = 5.0,
        download_timeout_seconds: float = 15.0,
        download_workers: int = 8,
    ) -> None:
        if check_timeout_seconds <= 0 or download_timeout_seconds <= 0:
            raise ValueError("network timeouts must be positive")
        if not 1 <= download_workers <= 16:
            raise ValueError("download_workers must be from 1 to 16")
        self.check_timeout_seconds = check_timeout_seconds
        self.download_timeout_seconds = download_timeout_seconds
        self.download_workers = download_workers

    @staticmethod
    def _request_bytes(url: str, *, timeout: float, maximum: int) -> bytes:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "Gakumas-Helper-arena-component-updater",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        last_error: Exception | None = None
        attempts_made = 0
        for attempt in range(1, HTTP_ATTEMPTS + 1):
            attempts_made = attempt
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = response.read(maximum + 1)
                if len(data) > maximum:
                    raise ArenaComponentError(f"RIS response exceeds the accepted size: {url}")
                return data
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code < 500 and error.code not in {408, 429}:
                    break
            except (OSError, urllib.error.URLError) as error:
                last_error = error
            if attempt < HTTP_ATTEMPTS:
                time.sleep(0.25 * attempt)
        failure = ArenaComponentError(
            f"RIS request failed after {attempts_made} attempt(s): {url}: {last_error}"
        )
        failure._arena_component_retry_exhausted = True
        raise failure from last_error

    @classmethod
    def _request_json(cls, url: str, *, timeout: float) -> object:
        data = cls._request_bytes(url, timeout=timeout, maximum=MAX_API_RESPONSE_BYTES)
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArenaComponentError(f"RIS response is not valid JSON: {url}") from error

    def discover_production_commit(self) -> str:
        deployments = self._request_json(
            f"{API_BASE}/deployments?environment=production&per_page={MAX_PRODUCTION_DEPLOYMENTS}",
            timeout=self.check_timeout_seconds,
        )
        if not isinstance(deployments, list):
            raise ArenaComponentError("RIS production deployment response is invalid")
        production_deployments = sorted(
            (
                deployment
                for deployment in deployments
                if isinstance(deployment, Mapping)
                and deployment.get("environment") == "production"
                and isinstance(deployment.get("id"), int)
                and not isinstance(deployment.get("id"), bool)
            ),
            key=lambda deployment: int(deployment["id"]),
            reverse=True,
        )
        for deployment in production_deployments:
            deployment_id = deployment.get("id")
            commit = deployment.get("sha")
            if (
                isinstance(deployment_id, bool)
                or not isinstance(deployment_id, int)
                or not isinstance(commit, str)
                or COMMIT_PATTERN.fullmatch(commit) is None
            ):
                continue
            statuses = self._request_json(
                f"{API_BASE}/deployments/{deployment_id}/statuses?per_page=1",
                timeout=self.check_timeout_seconds,
            )
            valid_statuses = (
                [
                    status
                    for status in statuses
                    if isinstance(status, Mapping)
                    and isinstance(status.get("id"), int)
                    and not isinstance(status.get("id"), bool)
                ]
                if isinstance(statuses, list)
                else []
            )
            latest_status = max(
                valid_statuses,
                key=lambda status: int(status["id"]),
                default=None,
            )
            if isinstance(latest_status, Mapping) and latest_status.get("state") == "success":
                return commit
        raise ArenaComponentError("RIS has no successful production deployment in the checked window")

    def fetch_stage_rows(self, commit: str) -> object:
        _validate_commit(commit)
        url = f"{RAW_BASE}/{commit}/packages/gakumas-data/json/stages.json"
        return self._request_json(url, timeout=self.download_timeout_seconds)

    def materialize_source(self, commit: str, destination: Path) -> None:
        _validate_commit(commit)
        if destination.exists() and any(destination.iterdir()):
            raise ArenaComponentError(f"RIS source destination must be absent or empty: {destination}")
        git_commit = self._request_json(
            f"{API_BASE}/git/commits/{commit}",
            timeout=self.download_timeout_seconds,
        )
        tree = git_commit.get("tree") if isinstance(git_commit, Mapping) else None
        tree_sha = tree.get("sha") if isinstance(tree, Mapping) else None
        if not isinstance(tree_sha, str) or COMMIT_PATTERN.fullmatch(tree_sha) is None:
            raise ArenaComponentError("RIS production commit does not expose a valid source tree")
        tree_payload = self._request_json(
            f"{API_BASE}/git/trees/{tree_sha}?recursive=1",
            timeout=self.download_timeout_seconds,
        )
        if not isinstance(tree_payload, Mapping) or tree_payload.get("truncated") is True:
            raise ArenaComponentError("RIS source tree is unavailable or truncated")
        entries = tree_payload.get("tree")
        if not isinstance(entries, list):
            raise ArenaComponentError("RIS source tree entries are invalid")

        files: dict[str, int] = {}
        for entry in entries:
            if not isinstance(entry, Mapping) or entry.get("type") != "blob":
                continue
            path = entry.get("path")
            size = entry.get("size")
            if not isinstance(path, str):
                continue
            if path != "LICENSE" and not path.startswith(SOURCE_PREFIXES):
                continue
            relative = _safe_source_relative_path(path)
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ArenaComponentError(f"RIS source file size is invalid: {path}")
            if size > MAX_SOURCE_FILE_BYTES:
                raise ArenaComponentError(f"RIS source file is unexpectedly large: {path}")
            files[relative.as_posix()] = size
        unique_paths = tuple(files)
        if len(unique_paths) > MAX_SOURCE_FILES:
            raise ArenaComponentError("RIS source tree contains too many runtime package files")
        if sum(files.values()) > MAX_SOURCE_TOTAL_BYTES:
            raise ArenaComponentError("RIS runtime package source exceeds the accepted total size")
        missing = REQUIRED_SOURCE_PATHS.difference(unique_paths)
        if missing:
            raise ArenaComponentError(f"RIS source tree is missing required files: {sorted(missing)}")

        destination.mkdir(parents=True, exist_ok=True)
        destination_root = destination.resolve()
        try:
            with ThreadPoolExecutor(max_workers=self.download_workers) as executor:
                futures = {
                    executor.submit(self._download_source_file, commit, path, files[path]): path
                    for path in unique_paths
                }
                for future in as_completed(futures):
                    path = futures[future]
                    data = future.result()
                    target = destination.joinpath(*PurePosixPath(path).parts).resolve(strict=False)
                    try:
                        target.relative_to(destination_root)
                    except ValueError as error:
                        raise ArenaComponentError(
                            f"unsafe RIS source target escaped the candidate root: {path!r}"
                        ) from error
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise

    def _download_source_file(self, commit: str, path: str, expected_size: int) -> bytes:
        encoded_path = urllib.parse.quote(path, safe="/")
        data = self._request_bytes(
            f"{RAW_BASE}/{commit}/{encoded_path}",
            timeout=self.download_timeout_seconds,
            maximum=MAX_SOURCE_FILE_BYTES,
        )
        if len(data) != expected_size:
            raise ArenaComponentError(
                f"RIS source file size changed for immutable commit {commit[:12]}: {path}"
            )
        return data


class ArenaComponentManager:
    """Cache successful checks and bound automatic recovery of failed updates."""

    def __init__(
        self,
        *,
        cache_root: Path = DEFAULT_COMPONENT_ROOT,
        source: ArenaUpstreamSource | None = None,
        host_revision: str | None = None,
        builder: Callable[..., None] = build_runtime_component,
        validator: Callable[[Path, str], _BundleInfo] | None = None,
    ) -> None:
        self.cache_root = cache_root.resolve()
        self.source = source or GitHubProductionSource()
        self.host_revision = host_revision or _detect_host_revision()
        self.builder = builder
        self._uses_default_validator = validator is None
        self.validator = validator or _validate_component_bundle
        self._lock = threading.Lock()
        self._checks: dict[Path, _CheckOutcome] = {}

    def reset_failed_checks(self) -> None:
        """Start a new task's failure budget while retaining successful checks."""

        with self._lock:
            self._checks = {path: outcome for path, outcome in self._checks.items() if outcome.failure is None}

    def resolve(
        self,
        baseline_bundle: Path,
        selection: str | int,
        *,
        check_updates: bool = True,
    ) -> ArenaComponentResolution:
        baseline = baseline_bundle.resolve()
        if not check_updates:
            info = _inspect_bundle(baseline)
            return _resolve_selection(
                _CheckOutcome(
                    bundle=info,
                    remote_commit=None,
                    remote_latest_season=None,
                    status="custom_bundle",
                    update_activated=False,
                    warnings=(),
                ),
                selection,
            )
        with self._lock:
            outcome = self._checks.get(baseline)
            if outcome is None:
                outcome = self._check_once(baseline)
                for _ in range(FAILED_CHECK_RETRIES):
                    if outcome.failure is None or not _recoverable_component_failure(outcome.failure):
                        break
                    outcome = self._check_once(baseline)
                self._checks[baseline] = outcome
        return _resolve_selection(outcome, selection)

    def _check_once(self, baseline_bundle: Path) -> _CheckOutcome:
        baseline = _inspect_bundle(baseline_bundle)
        active = self._load_active()
        if active is None:
            active = baseline
        active_reference_warnings = (
            _validate_component_reference_coverage(
                active.path,
                enforce_p_item_reverse_compatibility=False,
            )
            if self._uses_default_validator
            else active.reference_warnings
        )
        warnings = list(active_reference_warnings)
        try:
            remote_commit = self.source.discover_production_commit()
        except Exception as error:
            _log_component_failure("discover_production", error, active_commit=active.commit)
            warnings.append(f"RIS 生产版本检查失败，继续使用 {active.commit[:12]}：{error}")
            return _CheckOutcome(
                bundle=active,
                remote_commit=None,
                remote_latest_season=None,
                status="check_failed_using_active",
                update_activated=False,
                warnings=tuple(warnings),
                failure=error,
            )

        if remote_commit == active.commit:
            return _CheckOutcome(
                bundle=active,
                remote_commit=remote_commit,
                remote_latest_season=active.latest_season,
                status="current",
                update_activated=False,
                warnings=tuple(warnings),
            )

        try:
            remote_catalog = ContestStageCatalog.from_rows(self.source.fetch_stage_rows(remote_commit))
            remote_latest = remote_catalog.resolve("latest").season
        except Exception as error:
            _log_component_failure("fetch_stage_catalog", error, commit=remote_commit)
            warnings.append(
                f"RIS 生产版本 {remote_commit[:12]} 的场地目录读取失败，继续使用 {active.commit[:12]}：{error}"
            )
            return _CheckOutcome(
                bundle=active,
                remote_commit=remote_commit,
                remote_latest_season=None,
                status="catalog_check_failed_using_active",
                update_activated=False,
                warnings=tuple(warnings),
                failure=error,
            )

        try:
            installed = self._install(remote_commit, baseline_bundle, expected_latest_season=remote_latest)
        except Exception as error:
            p_item_coverage_update_failed = isinstance(
                error,
                ArenaPItemCoverageError,
            )
            warnings.append(
                f"RIS 竞技场组件更新到 {remote_commit[:12]} 失败，继续使用 {active.commit[:12]}：{error}"
            )
            return _CheckOutcome(
                bundle=active,
                remote_commit=remote_commit,
                remote_latest_season=remote_latest,
                status="update_failed_using_active",
                update_activated=False,
                warnings=tuple(warnings),
                p_item_coverage_update_failed=p_item_coverage_update_failed,
                failure=error,
            )

        return _CheckOutcome(
            bundle=installed,
            remote_commit=remote_commit,
            remote_latest_season=remote_latest,
            status="updated",
            update_activated=True,
            warnings=installed.reference_warnings,
        )

    def _load_active(self) -> _BundleInfo | None:
        state_path = self.cache_root / "active.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        commit = state.get("commit") if isinstance(state, Mapping) else None
        version_key = state.get("version_key") if isinstance(state, Mapping) else None
        if (
            not isinstance(state, Mapping)
            or state.get("schema_version") != STATE_SCHEMA_VERSION
            or state.get("host_revision") != self.host_revision
            or not isinstance(commit, str)
            or COMMIT_PATTERN.fullmatch(commit) is None
            or not isinstance(version_key, str)
            or version_key != _component_version_key(commit, self.host_revision)
        ):
            return None
        candidate = self.cache_root / "versions" / version_key
        try:
            info = _inspect_bundle(candidate)
        except ArenaComponentError:
            return None
        if info.commit != commit or info.host_revision != self.host_revision:
            return None
        return info

    def _install(
        self, commit: str, baseline_bundle: Path, *, expected_latest_season: int | None = None,
    ) -> _BundleInfo:
        versions = self.cache_root / "versions"
        version_key = _component_version_key(commit, self.host_revision)
        destination = versions / version_key
        stage = "cached_validation"

        def validate_identity(info: _BundleInfo) -> None:
            if info.host_revision != self.host_revision:
                raise ArenaComponentError("component host revision is incompatible")
            if expected_latest_season is not None and info.latest_season != expected_latest_season:
                raise ArenaComponentError("component stage catalog differs from the immutable production catalog")

        try:
            if destination.exists():
                try:
                    installed = self.validator(destination, commit)
                except ArenaPItemCoverageError:
                    raise
                except Exception as error:
                    raise ArenaComponentError(
                        "cached component directory is invalid; the immutable version was not overwritten"
                    ) from error
                validate_identity(installed)
            else:
                stage = "create_temporary_directory"
                self.cache_root.mkdir(parents=True, exist_ok=True)
                versions.mkdir(parents=True, exist_ok=True)
                temporary = tempfile.TemporaryDirectory(prefix=".candidate-", dir=self.cache_root)
                primary_error: BaseException | None = None
                try:
                    source_root = Path(temporary.name) / "source"
                    output = Path(temporary.name) / "bundle"
                    stage = "materialize_source"
                    self.source.materialize_source(commit, source_root)
                    stage = "build"
                    self.builder(
                        source_root, output, baseline_bundle,
                        upstream_commit=commit, host_revision=self.host_revision,
                    )
                    stage = "validate_candidate"
                    candidate = self.validator(output, commit)
                    validate_identity(candidate)
                    stage = "promote"
                    try:
                        _promote_component_directory(output, destination)
                    except FileExistsError:
                        # A concurrent immutable version is validated below, never overwritten.
                        pass
                    stage = "validate_installed"
                    installed = self.validator(destination, commit)
                    validate_identity(installed)
                except BaseException as error:
                    primary_error = error
                    raise
                finally:
                    try:
                        temporary.cleanup()
                    except Exception as cleanup_error:
                        if primary_error is None:
                            stage = "cleanup"
                            raise
                        primary_error.add_note(f"Temporary cleanup also failed: {cleanup_error!r}")
                        _log_component_failure(
                            "cleanup_secondary", cleanup_error, commit=commit,
                            primary_stage=stage, temporary=temporary.name,
                        )
            stage = "activate"
            self._activate(installed)
            return installed
        except Exception as error:
            error.add_note(f"RIS component install stage={stage} commit={commit}")
            _log_component_failure(stage, error, commit=commit, destination=destination)
            raise

    def _activate(self, bundle: _BundleInfo) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "commit": bundle.commit,
            "host_revision": self.host_revision,
            "version_key": _component_version_key(bundle.commit, self.host_revision),
        }
        temporary = self.cache_root / f"active.{uuid.uuid4().hex}.tmp"
        primary_error: BaseException | None = None
        try:
            temporary.write_text(
                json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.cache_root / "active.json")
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"Activation temporary cleanup also failed: {cleanup_error!r}")
                _log_component_failure("activate_cleanup_secondary", cleanup_error, temporary=temporary)


def _validate_commit(commit: str) -> None:
    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise ArenaComponentError("RIS commit must be a lowercase 40-character SHA")


def _safe_source_relative_path(value: str) -> PurePosixPath:
    if "\\" in value or ":" in value or "\x00" in value:
        raise ArenaComponentError(f"unsafe RIS source path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ArenaComponentError(f"unsafe RIS source path: {value!r}")
    return path


def _component_version_key(commit: str, host_revision: str) -> str:
    """Namespace immutable bundles by both RIS and GKH runtime identity."""

    _validate_commit(commit)
    host_key = hashlib.sha256(host_revision.encode("utf-8")).hexdigest()[:16]
    return f"{commit}-{host_key}"


def _read_manifest(bundle: Path) -> Mapping[str, Any]:
    path = bundle / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArenaComponentError(f"arena component manifest is unavailable or invalid: {path}") from error
    if not isinstance(value, Mapping):
        raise ArenaComponentError("arena component manifest root must be an object")
    return value


def _inspect_bundle(bundle: Path) -> _BundleInfo:
    manifest = _read_manifest(bundle)
    commit = manifest.get("commit")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("project") != "gakumas-tools"
        or manifest.get("adapter_protocol_version") != PROTOCOL_VERSION
        or not isinstance(commit, str)
        or COMMIT_PATTERN.fullmatch(commit) is None
    ):
        raise ArenaComponentError("arena component manifest identity or protocol is invalid")
    for name in ("node.exe", "runner.mjs", "scoring.mjs"):
        if not (bundle / name).is_file():
            raise ArenaComponentError(f"arena component runtime file is missing: {name}")
    try:
        catalog = ContestStageCatalog.from_bundle(bundle)
        latest = catalog.resolve("latest").season
    except StageCatalogError as error:
        raise ArenaComponentError(f"arena component stage catalog is invalid: {error}") from error
    component_update = manifest.get("component_update")
    host_revision = (
        component_update.get("host_revision")
        if isinstance(component_update, Mapping)
        and component_update.get("schema_version") == COMPONENT_UPDATE_SCHEMA_VERSION
        else None
    )
    if host_revision is not None and not isinstance(host_revision, str):
        raise ArenaComponentError("arena component host revision is invalid")
    return _BundleInfo(bundle.resolve(), commit, catalog, latest, host_revision)


def _validate_component_bundle(bundle: Path, expected_commit: str) -> _BundleInfo:
    info = _inspect_bundle(bundle)
    if info.commit != expected_commit:
        raise ArenaComponentError("arena component commit does not match the requested production deployment")
    reference_warnings = _validate_component_reference_coverage(bundle)

    stage_ids = [
        stage.stage_id
        for season in info.catalog.seasons
        for stage in info.catalog.resolve(season).stages
    ]
    smoke_path = bundle / f".component-smoke-{uuid.uuid4().hex}.mjs"
    smoke_path.write_text(
        "\n".join(
            (
                'import { StageConfig } from "gakumas-engine";',
                'import { Stages } from "gakumas-data";',
                'import { compareStageScores, countOutcomes } from "./scoring.mjs";',
                f"const ids = {json.dumps(stage_ids)};",
                "for (const id of ids) {",
                "  const stage = Stages.getById(id);",
                '  if (!stage || stage.type !== "contest") throw new Error(`missing contest stage ${id}`);',
                "  new StageConfig(stage);",
                "}",
                'if (typeof compareStageScores !== "function" || typeof countOutcomes !== "function") throw new Error("scoring exports missing");',
                'process.stdout.write("ok");',
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    primary_error: BaseException | None = None
    try:
        try:
            completed = subprocess.run(
                (bundle / "node.exe", smoke_path),
                cwd=bundle,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
                check=False,
                creationflags=creation_flags,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ArenaComponentError(f"arena component import smoke could not complete: {error}") from error
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            smoke_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(f"Import smoke cleanup also failed: {cleanup_error!r}")
            _log_component_failure("smoke_cleanup_secondary", cleanup_error, temporary=smoke_path)
    if completed.returncode != 0 or completed.stdout != "ok":
        detail = (completed.stderr or completed.stdout).strip()[-1000:]
        raise ArenaComponentError(f"arena component import smoke failed: {detail}")
    _validate_runner_protocol(bundle, info)
    return _BundleInfo(
        info.path,
        info.commit,
        info.catalog,
        info.latest_season,
        info.host_revision,
        reference_warnings,
    )


def _validate_component_reference_coverage(
    bundle: Path,
    *,
    enforce_p_item_reverse_compatibility: bool = True,
) -> tuple[str, ...]:
    """Validate host galleries and warn when RIS has newer recognition IDs."""

    try:
        entity_catalog = ArenaEntityCatalog.from_bundle(bundle)
    except (ArenaCatalogError, OSError, ValueError) as error:
        raise ArenaComponentError(f"arena component entity catalog is invalid: {error}") from error
    try:
        from p_item_recognition import PItemReferenceError, PItemRenderedReferenceGallery
    except ImportError as error:
        raise ArenaComponentError(
            "host P-item reference runtime is unavailable"
        ) from error
    try:
        reference_gallery = PItemRenderedReferenceGallery.load(
            _resolve_host_p_item_reference_root()
        )
    except (OSError, KeyError, TypeError, ValueError, PItemReferenceError) as error:
        raise ArenaComponentError(
            "host P-item reference gallery is unavailable or invalid"
        ) from error
    represented_p_item_ids = frozenset(
        int(value) for value in reference_gallery.p_item_ids
    )
    required_p_item_ids = frozenset(
        entity_catalog.arena_p_item_reference_required_ids()
    )
    missing_p_item_ids = tuple(
        sorted(required_p_item_ids - represented_p_item_ids)
    )
    if enforce_p_item_reverse_compatibility:
        _validate_component_p_item_coverage(
            entity_catalog,
            reference_gallery.p_item_ids,
            provisional_gallery_ids=reference_gallery.provisional_p_item_ids,
        )
    try:
        card_gallery = BadgeReferenceGallery.load(
            _resolve_host_card_reference_root()
        )
    except (OSError, KeyError, TypeError, ValueError, BadgeReferenceError) as error:
        raise ArenaComponentError(
            "host skill-card reference gallery is unavailable or invalid"
        ) from error
    missing_skill_card_ids = _validate_component_skill_card_coverage(
        entity_catalog,
        card_gallery.business_ids,
    )
    fixed_slot_skill_card_ids: list[int] = []
    gallery_update_required_skill_card_ids: list[int] = []
    for card_id in missing_skill_card_ids:
        try:
            source_type = entity_catalog.skill_card_source_type(card_id)
        except ArenaCatalogError:
            gallery_update_required_skill_card_ids.append(card_id)
            continue
        if source_type in {"pIdol", "support"}:
            fixed_slot_skill_card_ids.append(card_id)
        else:
            gallery_update_required_skill_card_ids.append(card_id)
    warnings: list[str] = []
    if missing_p_item_ids:
        warnings.append(
            "RIS_RECOGNITION_REFERENCE_LAG "
            f"kind=p_item ids={missing_p_item_ids!r} mode=global_detail"
        )
    if fixed_slot_skill_card_ids:
        warnings.append(
            "RIS_RECOGNITION_REFERENCE_LAG "
            f"kind=skill_card ids={tuple(fixed_slot_skill_card_ids)!r} "
            "mode=fixed_slot_exact_title_detail"
        )
    if gallery_update_required_skill_card_ids:
        warnings.append(
            "RIS_RECOGNITION_REFERENCE_LAG "
            "kind=skill_card "
            f"ids={tuple(gallery_update_required_skill_card_ids)!r} "
            "mode=gallery_update_required"
        )
    return tuple(warnings)


def _validate_component_p_item_coverage(
    catalog: ArenaEntityCatalog,
    gallery_ids: Sequence[int],
    *,
    provisional_gallery_ids: Sequence[int] = (),
) -> tuple[int, ...]:
    """Return forward drift while rejecting reverse P-item incompatibility."""

    represented = frozenset(int(value) for value in gallery_ids)
    required = frozenset(catalog.arena_p_item_reference_required_ids())
    missing = tuple(sorted(required - represented))
    provisional = frozenset(int(value) for value in provisional_gallery_ids)
    stale_provisional = tuple(sorted(provisional - required))
    if stale_provisional:
        raise ArenaPItemCoverageError(
            "host provisional P-item IDs are absent from the candidate arena-stage "
            f"catalog: {stale_provisional!r}"
        )
    catalog_ids = frozenset(catalog.p_item_business_ids())
    unknown_gallery_ids = tuple(sorted(represented - catalog_ids))
    if unknown_gallery_ids:
        raise ArenaPItemCoverageError(
            "host P-item reference gallery contains IDs absent from the candidate "
            f"P-item catalog: {unknown_gallery_ids!r}"
        )
    return missing


def _validate_component_skill_card_coverage(
    catalog: ArenaEntityCatalog,
    gallery_ids: Sequence[int],
) -> tuple[int, ...]:
    """Return skill-card IDs supplied by RIS but absent from the host gallery."""

    represented = frozenset(int(value) for value in gallery_ids)
    catalog_ids = frozenset(catalog.skill_card_business_ids())
    return tuple(sorted(catalog_ids - represented))


def _validate_runner_protocol(bundle: Path, info: _BundleInfo) -> None:
    """Execute the real adapter path with a cheap, deterministic empty loadout."""

    latest = info.catalog.resolve("latest")
    stage_ids = list(latest.stage_ids)
    loadout = {
        "params": [1, 1, 1, 1],
        "pItemIds": [0, 0, 0, 0],
        "skillCardIdGroups": [[0] * 6, [0] * 6],
        "customizationGroups": [[{} for _ in range(6)], [{} for _ in range(6)]],
    }
    request = {
        "schema_version": PROTOCOL_VERSION,
        "request_id": "arena-component-smoke",
        "operation": "arena_own_score",
        "simulations": 1000,
        "seed": 400,
        "season": latest.season,
        "stageIds": stage_ids,
        "score_aggregation": OWN_SCORE_AGGREGATION,
        "parallelism": {"mode": "fixed", "workers": 1},
        "own_team": {
            "team_id": "arena-component-smoke-own",
            "supportBonus": 0.0,
            "stages": [
                {
                    "stage_number": index,
                    "stageId": stage_id,
                    "members": [{"slot": 1, "loadout": loadout}],
                }
                for index, stage_id in enumerate(stage_ids, start=1)
            ],
        },
    }
    try:
        batch = SubprocessArenaAdapter.from_bundle(bundle, timeout_seconds=30).simulate_own(request)
    except (AdapterError, OSError, ValueError) as error:
        raise ArenaComponentError(f"arena component runner protocol smoke failed: {error}") from error
    if (
        batch.upstream_commit != info.commit
        or len(batch.stage_distributions) != 3
        or any(
            stage.get("stage_id") != expected
            for stage, expected in zip(batch.stage_distributions, stage_ids, strict=True)
        )
    ):
        raise ArenaComponentError("arena component runner protocol smoke returned inconsistent identity")


def _detect_host_revision() -> str:
    build_manifest = PROJECT_ROOT / "GAKUMAS_HELPER_BUILD.json"
    try:
        build = json.loads(build_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        build = None
    if isinstance(build, Mapping):
        source = build.get("source")
        revision = source.get("revision") if isinstance(source, Mapping) else None
        if isinstance(revision, str) and revision.strip():
            return f"source:{revision.strip()}"
        version = build.get("derived_version")
        if isinstance(version, str) and version.strip():
            return f"version:{version.strip()}"
    for interface_path in (PROJECT_ROOT / "interface.json", PROJECT_ROOT / "assets" / "interface.json"):
        try:
            interface = json.loads(interface_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        version = interface.get("version") if isinstance(interface, Mapping) else None
        if isinstance(version, str) and version.strip():
            return f"version:{version.strip()}"
    return "development"


def _resolve_selection(outcome: _CheckOutcome, selection: str | int) -> ArenaComponentResolution:
    try:
        season = outcome.bundle.catalog.resolve(selection)
    except StageCatalogError as error:
        raise ArenaComponentError(f"active arena component cannot resolve the selected season: {error}") from error
    if (
        outcome.p_item_coverage_update_failed
        and outcome.remote_latest_season is not None
        and season.season >= outcome.remote_latest_season
    ):
        raise ArenaComponentError(
            "RIS production P-item coverage changed, but its complete engine/data "
            "component could not be activated; the current/latest season will not use "
            "the older P-item catalog"
        ) from outcome.failure
    if (
        selection == "latest"
        and outcome.remote_latest_season is not None
        and outcome.remote_latest_season > outcome.bundle.latest_season
    ):
        raise ArenaComponentError(
            "RIS production has a newer arena season, but its complete engine/data component could not be activated; "
            "latest will not fall back to the older season"
        ) from outcome.failure
    return ArenaComponentResolution(
        bundle_dir=outcome.bundle.path,
        season=season,
        commit=outcome.bundle.commit,
        remote_commit=outcome.remote_commit,
        remote_latest_season=outcome.remote_latest_season,
        status=outcome.status,
        update_activated=outcome.update_activated,
        warnings=outcome.warnings,
    )


_DEFAULT_MANAGER = ArenaComponentManager()
_DEFAULT_SEASON_UI = ArenaSeasonUiSynchronizer(PROJECT_ROOT / "interface.json")


def _sync_default_season_ui(info: _BundleInfo) -> None:
    try:
        _DEFAULT_SEASON_UI.sync(
            info.catalog,
            (str(info.path), info.commit, _DEFAULT_MANAGER.host_revision),
            on_error=lambda error: _log_component_failure("season_ui_sync", error, component=info.path),
        )
    except Exception as error:
        _log_component_failure("season_ui_sync", error, component=info.path)


def sync_arena_season_ui_on_startup() -> None:
    """Use local activated metadata only; never check upstream or execute Node."""

    try:
        if not _DEFAULT_SEASON_UI.interface_path.is_file():
            return
        info = _DEFAULT_MANAGER._load_active() or _inspect_bundle(DEFAULT_BUNDLE_DIR)
        _sync_default_season_ui(info)
    except Exception as error:
        _log_component_failure("season_ui_startup", error)


def reset_failed_arena_component_checks() -> None:
    """Called once at the existing new-task entry; successful checks remain cached."""

    _DEFAULT_MANAGER.reset_failed_checks()
    _DEFAULT_SEASON_UI.reset_failed()


def resolve_arena_component(
    bundle_dir: Path,
    selection: str | int,
) -> ArenaComponentResolution:
    """Resolve one version-matched runtime bundle, checking RIS only for the default bundle."""

    baseline = bundle_dir.resolve()
    is_default = baseline == DEFAULT_BUNDLE_DIR.resolve()
    result = _DEFAULT_MANAGER.resolve(
        baseline,
        selection,
        check_updates=is_default,
    )
    if is_default:
        outcome = _DEFAULT_MANAGER._checks.get(baseline)
        if outcome is not None:
            _sync_default_season_ui(outcome.bundle)
    return result
