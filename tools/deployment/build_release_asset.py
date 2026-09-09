"""Create a deterministic full Maa release asset and its compatibility manifest.

The release packer accepts only a candidate already produced by
``build_derived_package.py`` for the project's own update repository.  It
revalidates the candidate before writing immutable GitHub Release assets.
"""

from __future__ import annotations

import os
import re
import json
import stat
import hashlib
import zipfile
import argparse
import tempfile
from typing import Any
from pathlib import Path
from email.parser import Parser
from collections.abc import Mapping

import numpy as np

try:
    from tools.deployment import build_derived_package as derived_package
except ModuleNotFoundError:  # Direct script execution from tools/deployment.
    import build_derived_package as derived_package

try:
    from tools.deployment.privacy_gate import PrivacyGateError, validate_tree
except ModuleNotFoundError:  # Direct script execution from tools/deployment.
    from privacy_gate import PrivacyGateError, validate_tree

try:
    from tools.deployment import promote_static_reference_handoff as static_handoff
    from tools.deployment.promote_static_reference_handoff import (
        StaticReferenceHandoffError,
        validate_promoted_component,
    )
except ModuleNotFoundError:  # Direct script execution from tools/deployment.
    import promote_static_reference_handoff as static_handoff
    from promote_static_reference_handoff import (
        StaticReferenceHandoffError,
        validate_promoted_component,
    )

SCHEMA_VERSION = 1
RELEASE_CHANNELS = {"arena_preview": "beta", "arena_release": "stable"}
PLATFORM = "win-x86_64"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

COMPONENT_MANIFESTS = {
    "card_vision": "resource/base/model/classify/card_embedding/manifest.json",
    "arena_card_vision": "resource/base/model/embedding/arena_card/manifest.json",
    "arena_badge_reference": "resource/base/model/embedding/arena_badge_reference/manifest.json",
    "arena_cost_reference": "resource/base/model/embedding/arena_card_cost_reference/manifest.json",
    "p_item_reference": "resource/base/model/embedding/p_item_reference/manifest.json",
    "arena_engine_catalog": "assets/arena-winrate/manifest.json",
}
RELEASE_READY_PRODUCTION_HANDOFF_STATUSES = frozenset(
    {"READY", "READY_WITH_REAL_SAMPLES_PENDING"}
)
NON_RELEASE_READY_PRODUCTION_HANDOFF_STATUSES = frozenset(
    {"PENDING_VALIDATION", "BLOCKED"}
)
RIS_ENGINE_PROMOTION_TOOL_CONTRACT = "task820-ris-engine-production-handoff-v1"
RIS_ENGINE_REBUILT_PAYLOAD_PREFIXES = (
    "node_modules/gakumas-engine/",
    "node_modules/gakumas-data/",
)
RIS_ENGINE_REBUILT_PAYLOAD_FILES = frozenset(
    {"THIRD_PARTY_NOTICES/gakumas-tools-LICENSE"}
)
RIS_ENGINE_REUSED_HOST_PAYLOAD_FILES = (
    "THIRD_PARTY_NOTICES/Node.js-LICENSE",
    "node.exe",
    "runner.mjs",
    "scoring.mjs",
)
GIT_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9A-Fa-f]{64}")
MFA_CORE_BUNDLE_SCHEMA_VERSION = 1
MFA_CORE_COMPONENT = "mfaavalonia_core"
MFA_CORE_STATUS = "READY"
MFA_CORE_BUNDLE_PAYLOAD_PATH = "MFAAvalonia.Core.dll"
MFA_CORE_INSTALL_PATH = "libs/MFAAvalonia.Core.dll"
MFA_CORE_NOTICE_PATH = "THIRD_PARTY_NOTICES/MFAAvalonia-LICENSE"
MFA_CORE_PATCH_PATH = (
    "tools/deployment/mfa/MFAAvalonia-v2.15.2-resource-update-github-fallback.patch"
)
MFA_CORE_UPSTREAM = {
    "repository": "https://github.com/MaaXYZ/MFAAvalonia",
    "tag": "v2.15.2",
    "commit": "6065fe33798b72906c5079fa6f210646801d9a5c",
    "source_archive": "MFAAvalonia-2.15.2.zip",
    "source_archive_sha256": "76DD02AFE4B1529B1D4F6416B3442E1BD7E64F13B72D8A3B428F955B5E26F67A",
    "version_checker_git_blob_sha1": "6e6d1118fa414ba21c7efa4f15a58ad95dd08bd7",
}
MFA_CORE_SDK_VERSION_COMPONENTS = (10, 0, 400)
MFA_CORE_SDK_VERSION = ".".join(str(part) for part in MFA_CORE_SDK_VERSION_COMPONENTS)
MFA_CORE_BUILD = {
    "sdk_version": MFA_CORE_SDK_VERSION,
    "sdk_archive": f"dotnet-sdk-{MFA_CORE_SDK_VERSION}-win-x64.zip",
    "sdk_archive_sha512": (
        "9B8B88590E4DA131BFD0DA7AA089D0FC04D5418D5F8607EC13D55DC5A17B4399"
        "AFD54D496C12657FA05C6C6546DC5EAB930F26AC6C50F2D3A7712C0FB378C366"
    ),
    "configuration": "Release",
    "runtime": "win-x64",
    "project": "MFAAvalonia/MFAAvalonia.csproj",
    "source_revision_id": "6065fe33798b72906c5079fa6f210646801d9a5c",
    "sentry_project_directory": "/_/MFAAvalonia/",
    "compiler_path_map_target": "/_/MFAAvalonia/",
    "project_directory_override_scope": "WriteSentryAttributes",
    "continuous_integration_build": True,
    "deterministic": True,
    "incremental_build": False,
    "max_cpu_count": 1,
    "sentry_cli": False,
}
MFA_CORE_PACKAGED_BUILD = {
    "sdk_version_components": list(MFA_CORE_SDK_VERSION_COMPONENTS),
    **{
        key: value
        for key, value in MFA_CORE_BUILD.items()
        if key not in {"sdk_version", "sdk_archive"}
    },
}
MFA_CORE_INPUT = {
    "baseline_dll_sha256": "2DF2226CE45FCE8AF0C4DF8022F399B0FA74E8378C9B23C4177BAC5EE6C295C8"
}
MFA_CORE_LICENSE = {
    "spdx": "GPL-3.0-only",
    "upstream_file": "LICENSE",
    "sha256": "3972DC9744F6499F0F9B2DBF76696F2AE7AD8AF9B23DDE66D6AF86C9DFB36986",
}
MFA_CORE_PATCH_SCOPE = [
    "resource_update_check",
    "resource_update_apply",
    "deterministic_build_path",
]
FORBIDDEN_DERIVED_UPDATE_KEYS = frozenset(
    {"mirrorchyan_rid", "mirrorchyan_multiplatform"}
)
BADGE_LIVE_APPEND_TOOL_CONTRACT = "task095-card-badge-live-reference-append-v1"
BADGE_LIVE_EVALUATION_TOOL_CONTRACT = (
    "task095-card-badge-hard-negative-evaluation-v1"
)
BADGE_LIVE_EVALUATION_SCHEMA_VERSION = 1
BADGE_LIVE_REQUIRED_EVALUATION_GATES = frozenset(
    {
        "append_only_five_array_prefix",
        "identity_sets_unchanged",
        "new_cross_group_exact_collision",
        "inherited_full_replay",
        "sibling_reference_rows",
        "baseline_wrong_identity_reproduced",
        "candidate_live_target",
        "fixed_geometry_stress",
        "authoritative_hard_negative_blind_test",
    }
)
BADGE_LIVE_REFERENCE_NAME_PATTERN = re.compile(r"L[0-9]{4}")
BADGE_LIVE_UPDATE_KEYS = frozenset(
    {
        "dataset_id",
        "dataset_revision",
        "source_kind",
        "identity_authority",
        "authorization_status",
        "raw_source_publication",
        "green_mask_source",
        "hard_negative_business_ids",
        "added_reference_count",
        "business_card_id_count",
        "business_card_ids",
        "visual_group_count",
        "visual_group_ids",
    }
)
FORBIDDEN_MUTABLE_PATHS = (
    ".local",
    "config",
    "logs",
    "cache",
    "captures",
    "screenshots",
    "profiles",
    "userdata",
    "user_data",
    "runtime-data",
    "temp",
    "tmp",
    "appsettings.json",
)
PROJECT_TEXT_ROOTS = {"agent", "resource", "tasks", "lang", "data"}
PROJECT_TEXT_FILES = {"README.md", "interface.json", "GAKUMAS_HELPER_BUILD.json", "requirements.txt"}
ARENA_PREVIEW_VERSION_PATTERN = re.compile(
    r"^v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$"
)
MINIMUM_INDEPENDENT_PROJECT_VERSION = "v0.1.0"
MINIMUM_P_ITEM_REFERENCE_BUSINESS_ID_COUNT = 426
P_ITEM_CATALOG_RELATIVE_PATH = "node_modules/gakumas-data/json/p_items.json"
SKILL_CARD_CATALOG_RELATIVE_PATH = "node_modules/gakumas-data/json/skill_cards.json"
P_ITEM_PRODUCTION_SOURCE_EVIDENCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "p-item-embedding"
    / "evidence"
)
FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION = "v0.1.1"
RESERVED_UNPUBLISHABLE_PROJECT_VERSIONS = frozenset({"v0.1.0"})
LAST_PUBLISHED_LEGACY_CHANNEL_VERSION = "v1.4.9+gkh.260823"
INTERNAL_TASK_IDENTIFIER_PATTERN = re.compile(rb"(?i)\bTA" rb"SK-[0-9]{3}\b")
RUNTIME_EMBEDDED_ARCHIVES = {
    "MaaAgentBinary/maatouch/universal/maatouch",
    "libs/MaaAgentBinary/maatouch/universal/maatouch",
    "libs/SharpCompress.dll",
    "python/Lib/site-packages/MaaAgentBinary/maatouch/universal/maatouch",
    "python/Scripts/pip.exe",
    "python/Scripts/pip3.exe",
    "python/Scripts/pip3.12.exe",
}


class ReleaseBuildError(RuntimeError):
    """Raised when a candidate is not safe to publish as one atomic release."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseBuildError(f"JSON is unavailable or invalid: {path}") from error
    if not isinstance(value, dict):
        raise ReleaseBuildError(f"JSON root must be an object: {path}")
    return value


def _select_p_item_production_source_evidence(
    component_root: Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Select one code-approved source proof using the bound report's digest."""

    evaluation = manifest.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("path") != static_handoff.REPORT_NAME
        or not static_handoff._is_sha256(evaluation.get("sha256"))
    ):
        raise StaticReferenceHandoffError(
            "p_item_reference evaluation report binding is invalid"
        )
    report_path = component_root / static_handoff.REPORT_NAME
    if (
        not report_path.is_file()
        or sha256_file(report_path).casefold()
        != str(evaluation["sha256"]).casefold()
    ):
        raise StaticReferenceHandoffError(
            "p_item_reference evaluation report hash mismatch"
        )
    report = static_handoff._load_object(
        report_path, label="p_item_reference evaluation report"
    )
    if report_path.read_bytes() != static_handoff.canonical_json_bytes(report):
        raise StaticReferenceHandoffError(
            "p_item_reference evaluation report is not canonical LF JSON"
        )
    promotion_evidence = report.get("promotion_evidence")
    evidence_sha256 = (
        promotion_evidence.get("production_source_evidence_sha256")
        if isinstance(promotion_evidence, Mapping)
        else None
    )
    if (
        not static_handoff._is_sha256(evidence_sha256)
        or str(evidence_sha256).upper()
        not in static_handoff.P_ITEM_PRODUCTION_SOURCE_EVIDENCE_SHA256_ALLOWLIST
    ):
        raise StaticReferenceHandoffError(
            "P-item production/source evidence digest is not in the code allowlist"
        )

    evidence_root = P_ITEM_PRODUCTION_SOURCE_EVIDENCE_DIR.resolve()
    matches: list[Path] = []
    try:
        for path in sorted(evidence_root.glob("p_item_production_source_*.json")):
            if path.resolve().parent != evidence_root:
                raise StaticReferenceHandoffError(
                    "P-item production/source evidence escapes the fixed directory"
                )
            if path.is_file() and sha256_file(path) == str(evidence_sha256).upper():
                matches.append(path)
    except OSError as error:
        raise StaticReferenceHandoffError(
            "P-item production/source evidence directory is unavailable"
        ) from error
    if len(matches) != 1:
        raise StaticReferenceHandoffError(
            "P-item production/source evidence requires exactly one fixed-directory "
            f"match; found {len(matches)}"
        )
    static_handoff._load_p_item_production_source_evidence(matches[0])
    return matches[0]


def _validate_component_production_handoff(
    component: str,
    manifest: Mapping[str, Any],
) -> None:
    if "production_handoff" not in manifest:
        raise ReleaseBuildError(
            f"component production handoff is missing: {component}"
        )
    handoff = manifest["production_handoff"]
    if not isinstance(handoff, Mapping):
        raise ReleaseBuildError(
            f"component production handoff is invalid: {component}"
        )
    status = handoff.get("status")
    known_statuses = (
        RELEASE_READY_PRODUCTION_HANDOFF_STATUSES
        | NON_RELEASE_READY_PRODUCTION_HANDOFF_STATUSES
    )
    if not isinstance(status, str) or status not in known_statuses:
        raise ReleaseBuildError(
            f"component production handoff status is invalid: {component}/{status}"
        )
    if status not in RELEASE_READY_PRODUCTION_HANDOFF_STATUSES:
        raise ReleaseBuildError(
            f"component production handoff is not release-ready: {component}/{status}"
        )


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _validate_ris_deployment_evidence(
    value: object,
    *,
    label: str,
    expected_revision: str,
    expected_ref: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseBuildError(f"arena engine {label} evidence is missing or invalid")
    if (
        value.get("revision") != expected_revision
        or value.get("environment") != "production"
        or value.get("ref") != expected_ref
        or value.get("task") != "deploy"
        or value.get("status") != "success"
        or not _positive_int(value.get("deployment_id"))
        or not _positive_int(value.get("status_id"))
        or any(
            not isinstance(value.get(field), str) or not value[field]
            for field in (
                "deployment_created_at",
                "deployment_updated_at",
                "status_created_at",
                "status_updated_at",
            )
        )
    ):
        raise ReleaseBuildError(f"arena engine {label} evidence is invalid")
    return value


def _validate_ris_engine_production_handoff(manifest: Mapping[str, Any]) -> None:
    """Reject a release engine unless its READY state is bound to promotion evidence."""

    handoff = manifest.get("production_handoff")
    if not isinstance(handoff, Mapping) or handoff.get("status") != "READY":
        raise ReleaseBuildError("arena engine requires a READY production handoff")
    if not isinstance(handoff.get("reason"), str) or not handoff["reason"]:
        raise ReleaseBuildError("arena engine production handoff reason is missing")

    commit = manifest.get("commit")
    branch = manifest.get("branch")
    source_archive_sha256 = manifest.get("source_archive_sha256")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("project") != "gakumas-tools"
        or manifest.get("repository")
        != "https://github.com/surisuririsu/gakumas-tools"
        or not isinstance(commit, str)
        or GIT_REVISION_PATTERN.fullmatch(commit) is None
        or not isinstance(branch, str)
        or not branch
        or not isinstance(source_archive_sha256, str)
        or SHA256_PATTERN.fullmatch(source_archive_sha256) is None
        or not isinstance(files, Mapping)
        or not files
    ):
        raise ReleaseBuildError("arena engine promotion source identity is invalid")

    promotion = manifest.get("promotion")
    if not isinstance(promotion, Mapping):
        raise ReleaseBuildError("arena engine promotion evidence is missing")
    if promotion.get("tool_contract") != RIS_ENGINE_PROMOTION_TOOL_CONTRACT:
        raise ReleaseBuildError("arena engine promotion tool contract is invalid")
    source_manifest_sha256 = promotion.get("source_manifest_sha256")
    source_manifest_canonical_sha256 = promotion.get(
        "source_manifest_canonical_sha256"
    )
    promoted_archive_sha256 = promotion.get("source_archive_sha256")
    source_projection = {
        key: value
        for key, value in manifest.items()
        if key not in {"production_handoff", "promotion"}
    }
    if (
        not isinstance(source_manifest_sha256, str)
        or SHA256_PATTERN.fullmatch(source_manifest_sha256) is None
        or not isinstance(source_manifest_canonical_sha256, str)
        or SHA256_PATTERN.fullmatch(source_manifest_canonical_sha256) is None
        or source_manifest_canonical_sha256.casefold()
        != _canonical_json_sha256(source_projection).casefold()
        or not isinstance(promoted_archive_sha256, str)
        or SHA256_PATTERN.fullmatch(promoted_archive_sha256) is None
        or promoted_archive_sha256.casefold() != source_archive_sha256.casefold()
    ):
        raise ReleaseBuildError("arena engine promotion source hashes are invalid")

    source_commit = promotion.get("source_commit")
    if not isinstance(source_commit, Mapping):
        raise ReleaseBuildError("arena engine source commit evidence is missing")
    parent_revisions = source_commit.get("parent_revisions")
    if (
        source_commit.get("revision") != commit
        or not isinstance(source_commit.get("tree_sha"), str)
        or GIT_REVISION_PATTERN.fullmatch(source_commit["tree_sha"]) is None
        or not isinstance(source_commit.get("committed_at"), str)
        or not source_commit["committed_at"]
        or not isinstance(parent_revisions, list)
        or not parent_revisions
        or any(
            not isinstance(parent, str)
            or GIT_REVISION_PATTERN.fullmatch(parent) is None
            for parent in parent_revisions
        )
        or len(parent_revisions) != len(set(parent_revisions))
    ):
        raise ReleaseBuildError("arena engine source commit evidence is invalid")

    production = _validate_ris_deployment_evidence(
        promotion.get("production_deployment"),
        label="production deployment",
        expected_revision=commit,
        expected_ref=branch,
    )
    rollback_revision = parent_revisions[0]
    rollback = _validate_ris_deployment_evidence(
        promotion.get("rollback_point"),
        label="rollback deployment",
        expected_revision=rollback_revision,
        expected_ref=branch,
    )
    if rollback["deployment_id"] >= production["deployment_id"]:
        raise ReleaseBuildError(
            "arena engine rollback deployment must precede the production deployment"
        )

    validation = promotion.get("validation")
    if not isinstance(validation, Mapping):
        raise ReleaseBuildError("arena engine promotion validation evidence is missing")
    declared_file_count = validation.get("declared_file_count")
    source_file_count = validation.get("source_archive_file_count")
    ris_rebuilt_file_count = validation.get("ris_rebuilt_payload_file_count")
    reused_files = validation.get("reused_host_payload_files")
    reused_file_count = validation.get("reused_host_payload_file_count")
    actual_ris_files = {
        str(relative)
        for relative in files
        if str(relative) in RIS_ENGINE_REBUILT_PAYLOAD_FILES
        or str(relative).startswith(RIS_ENGINE_REBUILT_PAYLOAD_PREFIXES)
    }
    actual_reused_files = tuple(sorted(set(map(str, files)) - actual_ris_files))
    if (
        not _positive_int(declared_file_count)
        or declared_file_count != len(files)
        or not _positive_int(source_file_count)
        or not _positive_int(ris_rebuilt_file_count)
        or source_file_count != ris_rebuilt_file_count
        or ris_rebuilt_file_count != len(actual_ris_files)
        or reused_files != list(RIS_ENGINE_REUSED_HOST_PAYLOAD_FILES)
        or reused_file_count != len(RIS_ENGINE_REUSED_HOST_PAYLOAD_FILES)
        or actual_reused_files != RIS_ENGINE_REUSED_HOST_PAYLOAD_FILES
        or declared_file_count != ris_rebuilt_file_count + reused_file_count
        or validation.get("source_archive_matches_git_tree") is not True
        or validation.get("ris_payload_rebuilt_from_source_archive") is not True
        or validation.get("payload_byte_identical") is not True
        or validation.get("node_import_and_stage_dsl") != "PASS"
        or validation.get("runner_protocol_smoke") != "PASS"
        or validation.get("adapter_protocol_version")
        != manifest.get("adapter_protocol_version")
        or validation.get("calibration_schema_version")
        != manifest.get("calibration_schema_version")
        or not _positive_int(validation.get("latest_complete_season"))
    ):
        raise ReleaseBuildError("arena engine promotion validation evidence is invalid")
    _positive_business_ids(
        validation.get("latest_stage_ids"),
        label="arena engine latest stage IDs",
    )
    if len(validation["latest_stage_ids"]) != 3:
        raise ReleaseBuildError("arena engine latest stage IDs must contain three values")
    runtime = manifest.get("runtime")
    runtime_executable_sha256 = (
        runtime.get("executable_sha256") if isinstance(runtime, Mapping) else None
    )
    payload_executable_sha256 = files.get("node.exe")
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(runtime_executable_sha256, str)
        or not isinstance(payload_executable_sha256, str)
        or runtime_executable_sha256.casefold()
        != payload_executable_sha256.casefold()
        or runtime.get("license_file")
        != "THIRD_PARTY_NOTICES/Node.js-LICENSE"
        or runtime["license_file"] not in files
    ):
        raise ReleaseBuildError("arena engine runtime provenance is not bound to its payload")


def _validate_ris_engine_stage_binding(
    engine_root: Path,
    manifest: Mapping[str, Any],
) -> None:
    rows = _load_array(
        _resolve_inside(engine_root, "node_modules/gakumas-data/json/stages.json")
    )
    by_season: dict[int, dict[int, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("type") != "contest":
            continue
        season = row.get("season")
        stage = row.get("stage")
        stage_id = row.get("id")
        if (
            not _positive_int(season)
            or stage not in (1, 2, 3)
            or isinstance(stage, bool)
            or not _positive_int(stage_id)
        ):
            raise ReleaseBuildError("arena engine contest stage catalog is invalid")
        stage_map = by_season.setdefault(season, {})
        if stage in stage_map:
            raise ReleaseBuildError("arena engine contest stage catalog contains duplicates")
        stage_map[stage] = stage_id
    if not by_season:
        raise ReleaseBuildError("arena engine contest stage catalog is empty")
    latest_season = max(by_season)
    latest = by_season[latest_season]
    if set(latest) != {1, 2, 3}:
        raise ReleaseBuildError("arena engine latest contest season is incomplete")
    validation = manifest["promotion"]["validation"]
    if (
        validation["latest_complete_season"] != latest_season
        or validation["latest_stage_ids"]
        != [latest[stage] for stage in (1, 2, 3)]
    ):
        raise ReleaseBuildError(
            "arena engine promotion season evidence disagrees with the catalog"
        )


def _validate_ris_engine_build_binding(
    build: Mapping[str, Any],
    engine_manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    engine = build.get("arena_engine")
    if (
        not isinstance(engine, Mapping)
        or engine.get("commit") != engine_manifest.get("commit")
        or not isinstance(engine.get("manifest_sha256"), str)
        or engine["manifest_sha256"].casefold()
        != sha256_file(manifest_path).casefold()
    ):
        raise ReleaseBuildError(
            "candidate build manifest is not bound to the arena engine manifest"
        )


def _validate_badge_live_evaluation(
    manifest_root: Path,
    manifest: Mapping[str, Any],
    *,
    arena_card_manifest: Mapping[str, Any],
    arena_reference_rows_by_business_id: Mapping[int, int],
    actual_reference_row_count: int,
    actual_live_reference_row_count: int,
    actual_business_id_count: int,
    actual_visual_group_count: int,
    actual_group_by_business_id: Mapping[int, str],
    actual_reference_rows_by_business_id: Mapping[int, int],
    actual_live_reference_rows: tuple[tuple[int, int, str], ...],
) -> None:
    build = manifest.get("build")
    source = manifest.get("source")
    live_updates = (
        source.get("live_reference_updates")
        if isinstance(source, Mapping)
        else None
    )
    requires_evaluation = (
        isinstance(build, Mapping)
        and build.get("tool_contract") == BADGE_LIVE_APPEND_TOOL_CONTRACT
    ) or bool(live_updates) or actual_live_reference_row_count > 0
    if not requires_evaluation:
        return

    if not isinstance(source, Mapping) or not isinstance(live_updates, list) or not live_updates:
        raise ReleaseBuildError(
            "arena badge live-reference lineage is missing or invalid"
        )
    if [index for index, _, _ in actual_live_reference_rows] != list(
        range(1, len(actual_live_reference_rows) + 1)
    ):
        raise ReleaseBuildError(
            "arena badge live-reference row names are not contiguous"
        )
    seen_revisions: set[tuple[str, str]] = set()
    total_live_reference_count = 0
    live_row_offset = 0
    for update in live_updates:
        if not isinstance(update, Mapping) or set(update) != BADGE_LIVE_UPDATE_KEYS:
            raise ReleaseBuildError(
                "arena badge live-reference lineage is missing or invalid"
            )
        dataset_id = update.get("dataset_id")
        dataset_revision = update.get("dataset_revision")
        added_count = update.get("added_reference_count")
        business_ids = update.get("business_card_ids")
        hard_negative_business_ids = update.get("hard_negative_business_ids")
        business_id_count = update.get("business_card_id_count")
        visual_groups = update.get("visual_group_ids")
        visual_group_count = update.get("visual_group_count")
        revision_key = (str(dataset_id), str(dataset_revision))
        if (
            not isinstance(dataset_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", dataset_id)
            is None
            or not isinstance(dataset_revision, str)
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
                dataset_revision,
            )
            is None
            or revision_key in seen_revisions
            or update.get("source_kind") != "AUTHORIZED_LOCAL_LIVE_CARD_CROP"
            or update.get("identity_authority")
            != "INDEPENDENT_DETAIL_TITLE_AND_FULL_EFFECT"
            or update.get("authorization_status")
            != "AUTHORIZED_FOR_LOCAL_DERIVATION"
            or update.get("raw_source_publication") != "EXCLUDED"
            or update.get("green_mask_source")
            != "SAME_VISUAL_GROUP_HISTORICAL_REFERENCE"
            or not isinstance(hard_negative_business_ids, list)
            or not hard_negative_business_ids
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in hard_negative_business_ids
            )
            or hard_negative_business_ids
            != sorted(set(hard_negative_business_ids))
            or isinstance(added_count, bool)
            or not isinstance(added_count, int)
            or added_count < 1
            or isinstance(business_id_count, bool)
            or not isinstance(business_id_count, int)
            or business_id_count != 1
            or not isinstance(business_ids, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in business_ids
            )
            or business_ids != sorted(set(business_ids))
            or len(business_ids) != business_id_count
            or set(business_ids).intersection(hard_negative_business_ids)
            or isinstance(visual_group_count, bool)
            or not isinstance(visual_group_count, int)
            or visual_group_count != 1
            or not isinstance(visual_groups, list)
            or any(not isinstance(value, str) or not value for value in visual_groups)
            or visual_groups != sorted(set(visual_groups))
            or len(visual_groups) != visual_group_count
            or added_count < business_id_count
            or visual_group_count > added_count
        ):
            raise ReleaseBuildError(
                "arena badge live-reference lineage is missing or invalid"
            )
        next_live_row_offset = live_row_offset + added_count
        batch_live_rows = actual_live_reference_rows[
            live_row_offset:next_live_row_offset
        ]
        batch_business_ids = sorted(
            {business_id for _, business_id, _ in batch_live_rows}
        )
        batch_visual_groups = sorted({group for _, _, group in batch_live_rows})
        if (
            batch_business_ids != business_ids
            or batch_visual_groups != visual_groups
            or not set(hard_negative_business_ids).issubset(
                actual_group_by_business_id
            )
        ):
            raise ReleaseBuildError(
                "arena badge live-reference lineage identity mapping disagrees "
                "with the gallery"
            )
        live_row_offset = next_live_row_offset
        seen_revisions.add(revision_key)
        total_live_reference_count += added_count
    if total_live_reference_count != actual_live_reference_row_count:
        raise ReleaseBuildError(
            "arena badge live-reference rows and lineage disagree"
        )
    last_update = live_updates[-1]
    reference_file_count = source.get("reference_file_count")
    added_reference_count = last_update.get("added_reference_count")
    if (
        isinstance(reference_file_count, bool)
        or not isinstance(reference_file_count, int)
        or reference_file_count != actual_reference_row_count
        or isinstance(added_reference_count, bool)
        or not isinstance(added_reference_count, int)
        or added_reference_count < 1
        or not isinstance(last_update.get("dataset_id"), str)
        or not last_update["dataset_id"]
        or not isinstance(last_update.get("dataset_revision"), str)
        or not last_update["dataset_revision"]
    ):
        raise ReleaseBuildError(
            "arena badge live-reference lineage disagrees with the gallery"
        )

    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ReleaseBuildError(
            "arena badge live-reference evaluation metadata is missing or invalid"
        )
    relative = evaluation.get("path")
    expected_hash = evaluation.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ReleaseBuildError(
            "arena badge live-reference evaluation path is invalid"
        )
    if (
        not isinstance(expected_hash, str)
        or re.fullmatch(r"[0-9A-Fa-f]{64}", expected_hash) is None
    ):
        raise ReleaseBuildError(
            "arena badge live-reference evaluation SHA-256 is invalid"
        )
    if evaluation.get("status") != "PASS":
        raise ReleaseBuildError(
            "arena badge live-reference evaluation status is not PASS"
        )
    if evaluation.get("tool_contract") != BADGE_LIVE_EVALUATION_TOOL_CONTRACT:
        raise ReleaseBuildError(
            "arena badge live-reference evaluation tool contract is invalid"
        )

    report_path = _resolve_inside(manifest_root, relative)
    if not report_path.is_file():
        raise ReleaseBuildError(
            "arena badge live-reference evaluation report is missing"
        )
    if sha256_file(report_path).casefold() != expected_hash.casefold():
        raise ReleaseBuildError(
            "arena badge live-reference evaluation report hash mismatch"
        )
    report = _load_object(report_path)
    report_schema_version = report.get("schema_version")
    if (
        isinstance(report_schema_version, bool)
        or report_schema_version != BADGE_LIVE_EVALUATION_SCHEMA_VERSION
    ):
        raise ReleaseBuildError(
            "arena badge live-reference evaluation report schema is invalid"
        )
    if report.get("status") != "PASS":
        raise ReleaseBuildError(
            "arena badge live-reference evaluation report status is not PASS"
        )
    if report.get("tool_contract") != BADGE_LIVE_EVALUATION_TOOL_CONTRACT:
        raise ReleaseBuildError(
            "arena badge live-reference evaluation report tool contract is invalid"
        )
    components = report.get("components")
    candidate = components.get("candidate") if isinstance(components, Mapping) else None
    report_gallery_hash = (
        candidate.get("gallery_sha256")
        if isinstance(candidate, Mapping)
        else None
    )
    gallery = manifest.get("gallery")
    manifest_gallery_hash = (
        gallery.get("sha256") if isinstance(gallery, Mapping) else None
    )
    if (
        not isinstance(report_gallery_hash, str)
        or not isinstance(manifest_gallery_hash, str)
        or report_gallery_hash.casefold() != manifest_gallery_hash.casefold()
    ):
        raise ReleaseBuildError(
            "arena badge live-reference evaluation report does not bind the manifest gallery"
        )

    candidate_reference_count = candidate.get("reference_row_count")
    if (
        isinstance(candidate_reference_count, bool)
        or not isinstance(candidate_reference_count, int)
        or candidate_reference_count != reference_file_count
        or candidate_reference_count != actual_reference_row_count
    ):
        raise ReleaseBuildError(
            "arena badge live-reference evaluation report row count is invalid"
        )
    base_component = (
        components.get("base") if isinstance(components, Mapping) else None
    )
    base_reference_count = (
        base_component.get("reference_row_count")
        if isinstance(base_component, Mapping)
        else None
    )
    declared_hashes = (
        candidate.get("manifest_sha256"),
        base_component.get("manifest_sha256")
        if isinstance(base_component, Mapping)
        else None,
        base_component.get("gallery_sha256")
        if isinstance(base_component, Mapping)
        else None,
    )
    if (
        isinstance(base_reference_count, bool)
        or not isinstance(base_reference_count, int)
        or base_reference_count < 1
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9A-Fa-f]{64}", value) is None
            for value in declared_hashes
        )
        or not isinstance(source.get("base_manifest_sha256"), str)
        or base_component["manifest_sha256"].casefold()
        != source["base_manifest_sha256"].casefold()
    ):
        raise ReleaseBuildError(
            "arena badge live-reference evaluation component provenance is invalid"
        )

    authoritative = report.get("authoritative_dataset")
    if (
        not isinstance(authoritative, Mapping)
        or not isinstance(authoritative.get("dataset_id"), str)
        or not authoritative["dataset_id"]
        or not isinstance(authoritative.get("dataset_revision"), str)
        or not authoritative["dataset_revision"]
        or not isinstance(authoritative.get("manifest_sha256"), str)
        or re.fullmatch(
            r"[0-9A-Fa-f]{64}",
            authoritative["manifest_sha256"],
        )
        is None
        or authoritative.get("source_domain")
        not in {"OFFICIAL_RAW_CARD_ART", "FIXED_RENDERED_PNG"}
        or authoritative.get("hard_negative_input")
        != "AUTHORITATIVE_FIXED_RENDERED_ICON"
    ):
        raise ReleaseBuildError(
            "arena badge live-reference authoritative dataset provenance is invalid"
        )
    arena_build = arena_card_manifest.get("build")
    arena_base_dataset_hash = (
        arena_build.get("base_dataset_manifest_sha256")
        if isinstance(arena_build, Mapping)
        else None
    )
    arena_base_dataset_lineage = (
        arena_build.get("base_dataset_lineage_manifest_sha256s")
        if isinstance(arena_build, Mapping)
        else None
    )
    if (
        not isinstance(arena_base_dataset_hash, str)
        or not isinstance(arena_base_dataset_lineage, list)
        or not arena_base_dataset_lineage
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9A-Fa-f]{64}", value) is None
            for value in arena_base_dataset_lineage
        )
        or len(arena_base_dataset_lineage)
        != len({value.casefold() for value in arena_base_dataset_lineage})
        or arena_base_dataset_hash.casefold()
        not in {value.casefold() for value in arena_base_dataset_lineage}
        or authoritative["manifest_sha256"].casefold()
        not in {value.casefold() for value in arena_base_dataset_lineage}
    ):
        raise ReleaseBuildError(
            "arena badge live-reference authoritative dataset is not bound "
            "to the arena card lineage"
        )

    live_batch = report.get("live_reference_batch")
    prototype_count = (
        live_batch.get("prototype_count")
        if isinstance(live_batch, Mapping)
        else None
    )
    if (
        not isinstance(live_batch, Mapping)
        or live_batch.get("dataset_id") != last_update["dataset_id"]
        or live_batch.get("dataset_revision") != last_update["dataset_revision"]
        or isinstance(prototype_count, bool)
        or not isinstance(prototype_count, int)
        or prototype_count != added_reference_count
        or live_batch.get("raw_source_publication") != "EXCLUDED"
    ):
        raise ReleaseBuildError(
            "arena badge live-reference evaluation report batch is invalid"
        )

    handoff = manifest.get("production_handoff")
    handoff_status = handoff.get("status") if isinstance(handoff, Mapping) else None
    holdout_status = report.get("independent_live_holdout")
    if (holdout_status, handoff_status) != (
        "PENDING",
        "READY_WITH_REAL_SAMPLES_PENDING",
    ):
        raise ReleaseBuildError(
            "arena badge live-reference holdout and production handoff disagree"
        )

    gates = report.get("gates")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != BADGE_LIVE_REQUIRED_EVALUATION_GATES
        or any(
            not isinstance(gate, Mapping) or gate.get("status") != "PASS"
            for gate in gates.values()
        )
    ):
        raise ReleaseBuildError(
            "arena badge live-reference evaluation report gates are incomplete"
        )

    def positive_count(gate_name: str, field: str = "sample_count") -> int | None:
        value = gates[gate_name].get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return None
        return value

    def has_zero_counts(gate_name: str, fields: tuple[str, ...]) -> bool:
        gate = gates[gate_name]
        return all(
            not isinstance(gate.get(field), bool)
            and isinstance(gate.get(field), int)
            and gate.get(field) == 0
            for field in fields
        )

    target_count = positive_count("candidate_live_target")
    baseline_count = positive_count("baseline_wrong_identity_reproduced")
    geometry_count = positive_count("fixed_geometry_stress")
    hard_negative_count = positive_count(
        "authoritative_hard_negative_blind_test"
    )
    inherited_count = positive_count("inherited_full_replay")
    append_inherited_count = positive_count(
        "append_only_five_array_prefix",
        "inherited_row_count",
    )
    append_prototype_count = positive_count(
        "append_only_five_array_prefix",
        "appended_prototype_count",
    )
    if (
        target_count != prototype_count
        or baseline_count != prototype_count
        or geometry_count is None
        or hard_negative_count is None
        or inherited_count is None
        or append_inherited_count is None
        or append_prototype_count != prototype_count
        or append_inherited_count + append_prototype_count
        != actual_reference_row_count
        or inherited_count != append_inherited_count
        or base_reference_count != append_inherited_count
        or geometry_count != 41 * prototype_count
        or gates["fixed_geometry_stress"].get("contract")
        != "dx=-3..3;dy=-1..1;size_delta=0,2;exact_excluded"
        or gates["identity_sets_unchanged"].get("business_id_count")
        != actual_business_id_count
        or gates["identity_sets_unchanged"].get("visual_group_count")
        != actual_visual_group_count
        or not has_zero_counts(
            "candidate_live_target",
            (
                "wrong_business_id_count",
                "wrong_visual_group_count",
                "low_confidence_count",
            ),
        )
        or not has_zero_counts(
            "fixed_geometry_stress",
            (
                "wrong_business_id_count",
                "wrong_visual_group_count",
                "low_confidence_count",
            ),
        )
        or not has_zero_counts(
            "authoritative_hard_negative_blind_test",
            (
                "wrong_business_id_count",
                "wrong_visual_group_count",
                "target_flip_count",
                "low_confidence_count",
            ),
        )
        or not has_zero_counts(
            "inherited_full_replay",
            (
                "candidate_changed_baseline_business_id_count",
                "wrong_visual_group_count",
            ),
        )
        or not has_zero_counts(
            "new_cross_group_exact_collision",
            ("collision_count",),
        )
    ):
        raise ReleaseBuildError(
            "arena badge live-reference evaluation report gate evidence is invalid"
        )

    scope = report.get("scope")
    hard_negative_ids = (
        scope.get("hard_negative_business_ids")
        if isinstance(scope, Mapping)
        else None
    )
    target_business_id = scope.get("target_business_id") if isinstance(scope, Mapping) else None
    target_visual_group_id = (
        scope.get("target_visual_group_id") if isinstance(scope, Mapping) else None
    )
    update_business_ids = last_update.get("business_card_ids")
    update_visual_group_ids = last_update.get("visual_group_ids")
    sibling_ids = scope.get("sibling_business_ids") if isinstance(scope, Mapping) else None
    baseline_wrong_id = (
        scope.get("baseline_wrong_business_id")
        if isinstance(scope, Mapping)
        else None
    )
    baseline_wrong_group = (
        scope.get("baseline_wrong_visual_group_id")
        if isinstance(scope, Mapping)
        else None
    )
    if (
        isinstance(target_business_id, bool)
        or not isinstance(target_business_id, int)
        or target_business_id < 1
        or not isinstance(target_visual_group_id, str)
        or not target_visual_group_id
        or not isinstance(update_business_ids, list)
        or update_business_ids != [target_business_id]
        or not isinstance(update_visual_group_ids, list)
        or update_visual_group_ids != [target_visual_group_id]
        or not isinstance(hard_negative_ids, list)
        or not hard_negative_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in hard_negative_ids
        )
        or target_business_id in hard_negative_ids
        or hard_negative_ids
        != sorted(
            {
                business_id
                for update in live_updates
                for business_id in update["hard_negative_business_ids"]
            }
        )
        or not isinstance(sibling_ids, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in sibling_ids
        )
        or sibling_ids != sorted(set(sibling_ids))
        or target_business_id in sibling_ids
        or set(sibling_ids).intersection(hard_negative_ids)
        or baseline_wrong_id not in hard_negative_ids
        or not isinstance(baseline_wrong_group, str)
        or not baseline_wrong_group
        or baseline_wrong_group == target_visual_group_id
    ):
        raise ReleaseBuildError(
            "arena badge live-reference evaluation report scope is invalid"
        )
    if (
        target_business_id not in actual_group_by_business_id
        or actual_group_by_business_id[target_business_id]
        != target_visual_group_id
        or baseline_wrong_id not in actual_group_by_business_id
        or actual_group_by_business_id[baseline_wrong_id]
        != baseline_wrong_group
        or any(
            business_id not in actual_group_by_business_id
            for business_id in sibling_ids
        )
        or any(
            business_id not in actual_group_by_business_id
            for business_id in hard_negative_ids
        )
        or scope.get("affected_visual_group_ids")
        != sorted(
            {
                actual_group_by_business_id[business_id]
                for business_id in {
                    target_business_id,
                    *sibling_ids,
                    *hard_negative_ids,
                }
            }
        )
    ):
        raise ReleaseBuildError(
            "arena badge live-reference evaluation report identity mapping "
            "disagrees with the gallery"
        )

    hard_negative_gate = gates["authoritative_hard_negative_blind_test"]
    per_business_count = hard_negative_gate.get("per_business_id_sample_count")
    expected_hard_negative_keys = {str(value) for value in hard_negative_ids}
    sibling_row_counts = gates["sibling_reference_rows"].get("row_counts")
    expected_sibling_keys = {str(value) for value in sibling_ids}
    expected_sibling_row_counts = {
        str(business_id): actual_reference_rows_by_business_id[business_id]
        for business_id in sibling_ids
    }
    if (
        not isinstance(per_business_count, Mapping)
        or set(per_business_count) != expected_hard_negative_keys
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in per_business_count.values()
        )
        or sum(per_business_count.values()) != hard_negative_count
        or any(
            arena_reference_rows_by_business_id.get(business_id)
            != per_business_count[str(business_id)]
            for business_id in hard_negative_ids
        )
        or not isinstance(sibling_row_counts, Mapping)
        or set(sibling_row_counts) != expected_sibling_keys
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in sibling_row_counts.values()
        )
        or dict(sibling_row_counts) != expected_sibling_row_counts
    ):
        raise ReleaseBuildError(
            "arena badge live-reference hard-negative evidence is invalid"
        )

    declared_scope_ids = {
        target_business_id,
        *sibling_ids,
        *hard_negative_ids,
    }
    expected_business_confusion = {
        business_id: actual_reference_rows_by_business_id[business_id]
        for business_id in declared_scope_ids
    }
    for business_id in hard_negative_ids:
        expected_business_confusion[business_id] += per_business_count[
            str(business_id)
        ]
    expected_visual_group_confusion: dict[str, int] = {}
    for business_id, count in expected_business_confusion.items():
        group = actual_group_by_business_id[business_id]
        expected_visual_group_confusion[group] = (
            expected_visual_group_confusion.get(group, 0) + count
        )

    confusion = report.get("confusion")
    business_sparse = (
        confusion.get("business_id_sparse")
        if isinstance(confusion, Mapping)
        else None
    )
    visual_group_sparse = (
        confusion.get("visual_group_sparse")
        if isinstance(confusion, Mapping)
        else None
    )
    if (
        not isinstance(confusion, Mapping)
        or set(confusion) != {"business_id_sparse", "visual_group_sparse"}
        or not isinstance(business_sparse, list)
        or not isinstance(visual_group_sparse, list)
    ):
        raise ReleaseBuildError(
            "arena badge live-reference evaluation confusion is missing or invalid"
        )

    actual_business_confusion: dict[int, int] = {}
    for row in business_sparse:
        if not isinstance(row, Mapping):
            raise ReleaseBuildError(
                "arena badge live-reference business confusion is invalid"
            )
        expected_id = row.get("expected_business_id")
        predicted_id = row.get("predicted_business_id")
        count = row.get("count")
        if (
            isinstance(expected_id, bool)
            or not isinstance(expected_id, int)
            or predicted_id != expected_id
            or expected_id not in expected_business_confusion
            or expected_id in actual_business_confusion
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise ReleaseBuildError(
                "arena badge live-reference business confusion is invalid"
            )
        actual_business_confusion[expected_id] = count
    if actual_business_confusion != expected_business_confusion:
        raise ReleaseBuildError(
            "arena badge live-reference business confusion counts are invalid"
        )

    actual_visual_group_confusion: dict[str, int] = {}
    for row in visual_group_sparse:
        if not isinstance(row, Mapping):
            raise ReleaseBuildError(
                "arena badge live-reference visual-group confusion is invalid"
            )
        expected_group = row.get("expected_visual_group_id")
        predicted_group = row.get("predicted_visual_group_id")
        count = row.get("count")
        if (
            not isinstance(expected_group, str)
            or not expected_group
            or predicted_group != expected_group
            or expected_group not in expected_visual_group_confusion
            or expected_group in actual_visual_group_confusion
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise ReleaseBuildError(
                "arena badge live-reference visual-group confusion is invalid"
            )
        actual_visual_group_confusion[expected_group] = count
    if actual_visual_group_confusion != expected_visual_group_confusion:
        raise ReleaseBuildError(
            "arena badge live-reference visual-group confusion counts are invalid"
        )

    evaluated_candidate_manifest = dict(manifest)
    evaluated_candidate_manifest.pop("evaluation", None)
    evaluated_candidate_manifest["production_handoff"] = {
        "status": "PENDING_VALIDATION",
        "reason": (
            "private-live candidate must pass append-only replay, target, geometry, "
            "and authoritative hard-negative evaluation before promotion"
        ),
    }
    evaluated_candidate_text = (
        json.dumps(
            evaluated_candidate_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    evaluated_candidate_bytes = evaluated_candidate_text.replace(
        "\n",
        os.linesep,
    ).encode("utf-8")
    evaluated_candidate_sha256 = hashlib.sha256(
        evaluated_candidate_bytes
    ).hexdigest()
    if (
        evaluated_candidate_sha256.casefold()
        != candidate["manifest_sha256"].casefold()
    ):
        raise ReleaseBuildError(
            "arena badge live-reference evaluation report is stale for the "
            "promoted manifest"
        )


def _load_array(path: Path) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseBuildError(f"JSON is unavailable or invalid: {path}") from error
    if not isinstance(value, list):
        raise ReleaseBuildError(f"JSON root must be an array: {path}")
    return value


def _positive_business_ids(values: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise ReleaseBuildError(f"{label} must contain positive integer business IDs")
    ids = tuple(values)
    if len(ids) != len(set(ids)):
        raise ReleaseBuildError(f"{label} must contain unique business IDs")
    return ids


def _load_p_item_gallery_ids(path: Path) -> tuple[int, ...]:
    try:
        with np.load(path, allow_pickle=False) as arrays:
            values = arrays["p_item_ids"]
            if values.ndim != 1 or values.dtype.kind not in {"i", "u"}:
                raise ReleaseBuildError(
                    "P-item reference gallery IDs must be a one-dimensional integer array"
                )
            raw_ids = values.tolist()
    except ReleaseBuildError:
        raise
    except (OSError, EOFError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise ReleaseBuildError("P-item reference gallery is unavailable or invalid") from error
    return _positive_business_ids(raw_ids, label="P-item reference gallery")


def _load_skill_card_gallery_contract(
    path: Path,
    *,
    label: str,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[str, str, str], ...],
    frozenset[int],
    frozenset[str],
    dict[int, str],
]:
    try:
        with np.load(path, allow_pickle=False) as arrays:
            required = {
                "embeddings",
                "class_names",
                "card_ids",
                "visual_group_ids",
                "upgrade_counts",
                "upgrade_markers",
            }
            if set(arrays.files) != required:
                raise ReleaseBuildError(
                    f"{label} gallery array inventory is incompatible"
                )
            embeddings = np.asarray(arrays["embeddings"])
            class_names = tuple(str(value) for value in arrays["class_names"].tolist())
            raw_card_ids = tuple(str(value) for value in arrays["card_ids"].tolist())
            visual_groups = tuple(str(value) for value in arrays["visual_group_ids"].tolist())
            upgrade_counts = np.asarray(arrays["upgrade_counts"])
            upgrade_markers = np.asarray(arrays["upgrade_markers"])
            if any(arrays[name].ndim != 1 for name in ("class_names", "card_ids", "visual_group_ids")):
                raise ReleaseBuildError(f"{label} gallery identity arrays must be one-dimensional")
    except ReleaseBuildError:
        raise
    except (OSError, EOFError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise ReleaseBuildError(f"{label} gallery is unavailable or invalid") from error
    if (
        not class_names
        or len(class_names) != len(raw_card_ids)
        or len(raw_card_ids) != len(visual_groups)
        or len(class_names) != len(set(class_names))
        or any(not value.isdigit() or int(value) < 1 for value in raw_card_ids)
        or any(not value for value in visual_groups)
        or any(name.split("_", 1)[0] != card_id for name, card_id in zip(class_names, raw_card_ids, strict=True))
        or embeddings.shape != (len(class_names), 128)
        or not np.issubdtype(embeddings.dtype, np.floating)
        or not np.isfinite(embeddings).all()
        or upgrade_counts.shape != (len(class_names),)
        or upgrade_counts.dtype.kind not in {"i", "u"}
        or not set(map(int, upgrade_counts)).issubset({0, 1})
        or upgrade_markers.shape != (len(class_names), 24, 24, 3)
        or not np.issubdtype(upgrade_markers.dtype, np.floating)
        or not np.isfinite(upgrade_markers).all()
    ):
        raise ReleaseBuildError(f"{label} gallery identity mapping is invalid")
    norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
    if np.any(norms <= 1e-8) or float(np.max(np.abs(norms - 1.0))) > 1e-3:
        raise ReleaseBuildError(f"{label} gallery embeddings are not L2-normalised")
    groups_by_id: dict[int, set[str]] = {}
    for raw_id, group in zip(raw_card_ids, visual_groups, strict=True):
        groups_by_id.setdefault(int(raw_id), set()).add(group)
    if any(len(groups) != 1 for groups in groups_by_id.values()):
        raise ReleaseBuildError(
            f"{label} gallery maps one business ID to multiple visual groups"
        )
    return (
        class_names,
        tuple(zip(class_names, raw_card_ids, visual_groups, strict=True)),
        frozenset(groups_by_id),
        frozenset(visual_groups),
        {business_id: next(iter(groups)) for business_id, groups in groups_by_id.items()},
    )


def _load_badge_gallery_contract(
    path: Path,
) -> tuple[
    frozenset[int],
    frozenset[str],
    dict[int, str],
    int,
    dict[int, int],
    tuple[tuple[int, int, str], ...],
]:
    try:
        with np.load(path, allow_pickle=False) as arrays:
            required = {
                "business_ids",
                "visual_group_ids",
                "reference_names",
                "top_coarse",
                "green_masks",
            }
            if set(arrays.files) != required:
                raise ReleaseBuildError(
                    "arena badge gallery array inventory is incompatible"
                )
            business_ids = arrays["business_ids"]
            visual_group_ids = arrays["visual_group_ids"]
            reference_names = arrays["reference_names"]
            top_coarse = arrays["top_coarse"]
            green_masks = arrays["green_masks"]
            count = len(business_ids)
            if (
                business_ids.ndim != 1
                or visual_group_ids.ndim != 1
                or business_ids.shape != visual_group_ids.shape
                or business_ids.dtype.kind not in {"i", "u"}
                or reference_names.shape != (count,)
                or len(set(map(str, reference_names))) != count
                or top_coarse.shape != (count, 8, 16, 3)
                or top_coarse.dtype != np.uint8
                or green_masks.shape != (count, 96, 96)
                or green_masks.dtype != np.uint8
                or not set(map(int, np.unique(green_masks))).issubset({0, 1})
            ):
                raise ReleaseBuildError("arena badge gallery identity arrays are invalid")
            raw_ids = business_ids.tolist()
            raw_groups = tuple(str(value) for value in visual_group_ids.tolist())
    except ReleaseBuildError:
        raise
    except (OSError, EOFError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise ReleaseBuildError("arena badge gallery is unavailable or invalid") from error
    if any(value < 1 for value in raw_ids) or any(not value for value in raw_groups):
        raise ReleaseBuildError("arena badge gallery identity mapping is invalid")
    groups_by_id: dict[int, set[str]] = {}
    row_counts_by_id: dict[int, int] = {}
    for business_id, group in zip(raw_ids, raw_groups, strict=True):
        groups_by_id.setdefault(business_id, set()).add(group)
        row_counts_by_id[business_id] = row_counts_by_id.get(business_id, 0) + 1
    if any(len(groups) != 1 for groups in groups_by_id.values()):
        raise ReleaseBuildError(
            "arena badge gallery maps one business ID to multiple visual groups"
        )
    live_rows = sorted(
        (
            int(str(name)[1:]),
            business_id,
            group,
        )
        for name, business_id, group in zip(
            reference_names,
            raw_ids,
            raw_groups,
            strict=True,
        )
        if BADGE_LIVE_REFERENCE_NAME_PATTERN.fullmatch(str(name)) is not None
    )
    return (
        frozenset(groups_by_id),
        frozenset(raw_groups),
        {business_id: next(iter(groups)) for business_id, groups in groups_by_id.items()},
        count,
        row_counts_by_id,
        tuple(live_rows),
    )


def _validate_skill_card_source_summary(
    source: object,
    *,
    actual_ids: frozenset[int],
    actual_groups: frozenset[str],
    label: str,
) -> None:
    if not isinstance(source, Mapping) or not actual_ids:
        raise ReleaseBuildError(f"{label} source metadata is invalid")
    minimum = source.get("business_card_id_min")
    maximum = source.get("business_card_id_max")
    if (minimum is None) != (maximum is None):
        raise ReleaseBuildError(f"{label} business-card range is incomplete")
    if minimum is not None:
        if (
            isinstance(minimum, bool)
            or isinstance(maximum, bool)
            or not isinstance(minimum, int)
            or not isinstance(maximum, int)
            or minimum < 1
            or maximum < minimum
            or actual_ids != frozenset(range(minimum, maximum + 1))
        ):
            raise ReleaseBuildError(f"{label} business-card range disagrees with the gallery")
    raw_ids = source.get("business_card_ids")
    if raw_ids is not None:
        declared_ids = frozenset(
            _positive_business_ids(raw_ids, label=f"{label} manifest")
        )
        if declared_ids != actual_ids:
            raise ReleaseBuildError(f"{label} exact business-card set disagrees with the gallery")
    raw_ranges = source.get("business_card_id_ranges")
    if raw_ranges is not None:
        if not isinstance(raw_ranges, list):
            raise ReleaseBuildError(f"{label} business-card ranges are invalid")
        declared_from_ranges: set[int] = set()
        for item in raw_ranges:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in item)
                or item[0] < 1
                or item[1] < item[0]
            ):
                raise ReleaseBuildError(f"{label} business-card ranges are invalid")
            values = set(range(item[0], item[1] + 1))
            if declared_from_ranges.intersection(values):
                raise ReleaseBuildError(f"{label} business-card ranges overlap")
            declared_from_ranges.update(values)
        if declared_from_ranges != actual_ids:
            raise ReleaseBuildError(f"{label} business-card ranges disagree with the gallery")
    if minimum is None and raw_ids is None and raw_ranges is None:
        raise ReleaseBuildError(f"{label} manifest has no exact business-card set")
    count = source.get("business_card_id_count")
    if count is not None and (
        isinstance(count, bool) or not isinstance(count, int) or count != len(actual_ids)
    ):
        raise ReleaseBuildError(f"{label} business-card count disagrees with the gallery")
    visual_count = source.get("visual_identity_count")
    if (
        isinstance(visual_count, bool)
        or not isinstance(visual_count, int)
        or visual_count != len(actual_groups)
    ):
        raise ReleaseBuildError(f"{label} visual-identity count disagrees with the gallery")


def _load_skill_card_catalog_ids(path: Path) -> frozenset[int]:
    rows = _load_array(path)
    ids: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReleaseBuildError("arena engine skill-card catalog row is invalid")
        value = row.get("id")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ReleaseBuildError("arena engine skill-card catalog ID is invalid")
        ids.append(value)
    if not ids or len(ids) != len(set(ids)):
        raise ReleaseBuildError("arena engine skill-card catalog IDs are empty or duplicated")
    return frozenset(ids)


def _pending_provisional_p_item_ids(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    source = manifest.get("source")
    if source is None:
        return ()
    if not isinstance(source, Mapping):
        raise ReleaseBuildError("P-item reference source metadata is invalid")
    provenance = source.get("extension_provenance")
    if provenance is None:
        return ()
    if not isinstance(provenance, Mapping):
        raise ReleaseBuildError("P-item reference extension provenance is invalid")
    validation = provenance.get("validation")
    if validation is None:
        return ()
    if not isinstance(validation, Mapping):
        raise ReleaseBuildError("P-item reference extension validation is invalid")
    pending = any(
        isinstance(key, str)
        and key.startswith("real_")
        and key.endswith("_sample")
        and value == "PENDING"
        for key, value in validation.items()
    )
    if not pending:
        return ()
    catalog = provenance.get("catalog")
    if not isinstance(catalog, Mapping):
        raise ReleaseBuildError(
            "pending P-item reference validation has no extension catalog"
        )
    ids = _positive_business_ids(
        catalog.get("included_business_ids"),
        label="pending P-item reference extension",
    )
    if not ids:
        raise ReleaseBuildError(
            "pending P-item reference validation has no included business IDs"
        )
    return ids


def _validate_p_item_release_contract(
    manifest: Mapping[str, Any],
    *,
    gallery_path: Path,
    catalog_path: Path,
) -> None:
    gallery = manifest.get("reference_gallery")
    if not isinstance(gallery, Mapping):
        raise ReleaseBuildError("P-item reference catalog is incompatible")
    declared_count = gallery.get("business_id_count")
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count < MINIMUM_P_ITEM_REFERENCE_BUSINESS_ID_COUNT
    ):
        raise ReleaseBuildError("P-item reference catalog is incompatible")

    gallery_ids = _load_p_item_gallery_ids(gallery_path)
    if len(gallery_ids) != declared_count:
        raise ReleaseBuildError(
            "P-item reference manifest count does not match the gallery ID set"
        )

    source = manifest.get("source")
    if source is not None:
        if not isinstance(source, Mapping):
            raise ReleaseBuildError("P-item reference source metadata is invalid")
        source_count = source.get("business_id_count", declared_count)
        if (
            isinstance(source_count, bool)
            or not isinstance(source_count, int)
            or source_count != declared_count
        ):
            raise ReleaseBuildError("P-item reference catalog is incompatible")

    provisional_ids = _positive_business_ids(
        gallery.get("provisional_business_ids", []),
        label="provisional P-item reference",
    )
    unknown_provisional = tuple(sorted(set(provisional_ids) - set(gallery_ids)))
    if unknown_provisional:
        raise ReleaseBuildError(
            "provisional P-item reference IDs are absent from the gallery: "
            f"{unknown_provisional!r}"
        )
    pending_ids = _pending_provisional_p_item_ids(manifest)
    undeclared_pending = tuple(sorted(set(pending_ids) - set(provisional_ids)))
    if undeclared_pending:
        raise ReleaseBuildError(
            "pending real-sample P-item IDs are not declared provisional: "
            f"{undeclared_pending!r}"
        )

    rows = _load_array(catalog_path)
    catalog_ids = []
    required_ids = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReleaseBuildError("arena P-item catalog rows must be objects")
        business_id = row.get("id")
        if isinstance(business_id, bool) or not isinstance(business_id, int) or business_id < 1:
            raise ReleaseBuildError("P-item catalog contains an invalid business ID")
        catalog_ids.append(business_id)
        if row.get("sourceType") not in {"pIdol", "support"} or row.get("mode") != "stage":
            continue
        required_ids.append(business_id)
    if len(catalog_ids) != len(set(catalog_ids)):
        raise ReleaseBuildError("P-item catalog contains duplicated business IDs")
    if not required_ids or len(required_ids) != len(set(required_ids)):
        raise ReleaseBuildError("arena P-item catalog required-ID set is empty or duplicated")
    required_id_set = set(required_ids)
    catalog_id_set = set(catalog_ids)
    unknown_gallery_ids = tuple(sorted(set(gallery_ids) - catalog_id_set))
    if unknown_gallery_ids:
        raise ReleaseBuildError(
            "P-item reference gallery contains IDs absent from the packaged P-item "
            f"catalog: {unknown_gallery_ids!r}"
        )
    missing = tuple(sorted(required_id_set - set(gallery_ids)))
    if missing:
        raise ReleaseBuildError(
            "P-item reference gallery does not cover arena-stage catalog IDs: "
            f"{missing!r}"
        )
    pending_outside_stage = tuple(sorted(set(pending_ids) - required_id_set))
    if pending_outside_stage:
        raise ReleaseBuildError(
            "pending real-sample P-item IDs are absent from the packaged arena-stage "
            f"catalog: {pending_outside_stage!r}"
        )
    provisional_outside_stage = tuple(
        sorted(set(provisional_ids) - required_id_set)
    )
    if provisional_outside_stage:
        raise ReleaseBuildError(
            "provisional P-item reference IDs are absent from the packaged arena-stage "
            f"catalog: {provisional_outside_stage!r}"
        )


def _resolve_inside(root: Path, relative_text: str) -> Path:
    relative = Path(relative_text.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ReleaseBuildError(f"candidate path escapes root: {relative_text}")
    candidate = (root / relative).resolve()
    if os.path.commonpath((str(root), str(candidate))) != str(root):
        raise ReleaseBuildError(f"candidate path escapes root: {relative_text}")
    return candidate


def _is_project_text(relative: Path) -> bool:
    parts = relative.parts
    return bool(parts) and (parts[0] in PROJECT_TEXT_ROOTS or relative.as_posix() in PROJECT_TEXT_FILES)


def _validate_release_privacy(candidate: Path) -> dict[str, int]:
    try:
        return validate_tree(
            candidate,
            project_path_predicate=_is_project_text,
            allowed_embedded_archives=RUNTIME_EMBEDDED_ARCHIVES,
        )
    except PrivacyGateError as error:
        raise ReleaseBuildError(str(error)) from error


def _validate_written_zip(
    candidate: Path,
    package: Path,
    *,
    expected_mfa_core_sha256: str,
) -> None:
    expected = {
        path.relative_to(candidate).as_posix(): (path.stat().st_size, sha256_file(path))
        for path in candidate.rglob("*")
        if path.is_file()
    }
    with zipfile.ZipFile(package) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ReleaseBuildError("release ZIP contains duplicate members")
        if set(names) != set(expected):
            raise ReleaseBuildError("release ZIP member inventory differs from the validated candidate")
        for entry in entries:
            relative = Path(entry.filename.replace("\\", "/"))
            mode = entry.external_attr >> 16
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode):
                raise ReleaseBuildError(f"release ZIP contains an unsafe member: {entry.filename}")
            expected_size, expected_hash = expected[entry.filename]
            payload = archive.read(entry)
            if len(payload) != expected_size or hashlib.sha256(payload).hexdigest().upper() != expected_hash:
                raise ReleaseBuildError(f"release ZIP member differs from candidate: {entry.filename}")
        packaged_mfa_core_sha256 = hashlib.sha256(
            archive.read(MFA_CORE_INSTALL_PATH)
        ).hexdigest().upper()
        if packaged_mfa_core_sha256.casefold() != expected_mfa_core_sha256.casefold():
            raise ReleaseBuildError(
                "release ZIP MFA Core payload differs from the validated provenance"
            )
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ReleaseBuildError(f"release ZIP integrity check failed: {corrupt}")


def _require_exact_mfa_object(
    value: object,
    keys: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReleaseBuildError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _validate_mfa_core_build_binding(
    candidate: Path,
    build: Mapping[str, Any],
    critical_files: Mapping[str, Any],
) -> None:
    mfa_core = _require_exact_mfa_object(
        build.get("mfa_core"),
        {
            "schema_version",
            "component",
            "status",
            "bundle_manifest_sha256",
            "upstream",
            "patch",
            "build",
            "input",
            "payload",
            "license",
        },
        label="candidate MFA Core provenance",
    )
    bundle_manifest_sha256 = mfa_core.get("bundle_manifest_sha256")
    if (
        mfa_core.get("schema_version") != MFA_CORE_BUNDLE_SCHEMA_VERSION
        or mfa_core.get("component") != MFA_CORE_COMPONENT
        or mfa_core.get("status") != MFA_CORE_STATUS
        or not isinstance(bundle_manifest_sha256, str)
        or SHA256_PATTERN.fullmatch(bundle_manifest_sha256) is None
    ):
        raise ReleaseBuildError("candidate MFA Core provenance identity is invalid")

    upstream = _require_exact_mfa_object(
        mfa_core.get("upstream"),
        set(MFA_CORE_UPSTREAM),
        label="candidate MFA Core upstream provenance",
    )
    if dict(upstream) != MFA_CORE_UPSTREAM:
        raise ReleaseBuildError(
            "candidate MFA Core provenance is not the fixed official v2.15.2 source"
        )

    patch = _require_exact_mfa_object(
        mfa_core.get("patch"),
        {"path", "sha256", "scope"},
        label="candidate MFA Core patch provenance",
    )
    if (
        patch.get("path") != MFA_CORE_PATCH_PATH
        or patch.get("scope") != MFA_CORE_PATCH_SCOPE
        or not isinstance(patch.get("sha256"), str)
        or SHA256_PATTERN.fullmatch(patch["sha256"]) is None
    ):
        raise ReleaseBuildError("candidate MFA Core patch provenance is invalid")

    build_provenance = _require_exact_mfa_object(
        mfa_core.get("build"),
        set(MFA_CORE_PACKAGED_BUILD),
        label="candidate MFA Core build provenance",
    )
    if dict(build_provenance) != MFA_CORE_PACKAGED_BUILD:
        raise ReleaseBuildError("candidate MFA Core build provenance is invalid")

    input_provenance = _require_exact_mfa_object(
        mfa_core.get("input"),
        set(MFA_CORE_INPUT),
        label="candidate MFA Core input provenance",
    )
    if dict(input_provenance) != MFA_CORE_INPUT:
        raise ReleaseBuildError("candidate MFA Core input provenance is invalid")

    payload = _require_exact_mfa_object(
        mfa_core.get("payload"),
        {"bundle_path", "install_path", "sha256", "size"},
        label="candidate MFA Core payload provenance",
    )
    payload_sha256 = payload.get("sha256")
    payload_size = payload.get("size")
    if (
        payload.get("bundle_path") != MFA_CORE_BUNDLE_PAYLOAD_PATH
        or payload.get("install_path") != MFA_CORE_INSTALL_PATH
        or not isinstance(payload_sha256, str)
        or SHA256_PATTERN.fullmatch(payload_sha256) is None
        or type(payload_size) is not int
        or payload_size <= 0
    ):
        raise ReleaseBuildError("candidate MFA Core payload provenance is invalid")

    license_provenance = _require_exact_mfa_object(
        mfa_core.get("license"),
        set(MFA_CORE_LICENSE),
        label="candidate MFA Core license provenance",
    )
    if dict(license_provenance) != MFA_CORE_LICENSE:
        raise ReleaseBuildError("candidate MFA Core license provenance is invalid")

    payload_path = _resolve_inside(candidate, MFA_CORE_INSTALL_PATH)
    if (
        not payload_path.is_file()
        or payload_path.stat().st_size != payload_size
        or sha256_file(payload_path).casefold() != payload_sha256.casefold()
    ):
        raise ReleaseBuildError("candidate MFA Core payload differs from its provenance")
    critical_payload_sha256 = critical_files.get(MFA_CORE_INSTALL_PATH)
    if (
        not isinstance(critical_payload_sha256, str)
        or critical_payload_sha256.casefold() != payload_sha256.casefold()
    ):
        raise ReleaseBuildError(
            "candidate MFA Core payload is not bound to the critical file inventory"
        )

    notice_path = _resolve_inside(candidate, MFA_CORE_NOTICE_PATH)
    if (
        not notice_path.is_file()
        or sha256_file(notice_path).casefold()
        != MFA_CORE_LICENSE["sha256"].casefold()
    ):
        raise ReleaseBuildError("candidate MFAAvalonia license notice is missing or invalid")
    critical_notice_sha256 = critical_files.get(MFA_CORE_NOTICE_PATH)
    if (
        not isinstance(critical_notice_sha256, str)
        or critical_notice_sha256.casefold()
        != MFA_CORE_LICENSE["sha256"].casefold()
    ):
        raise ReleaseBuildError(
            "candidate MFAAvalonia license notice is not bound to the critical file inventory"
        )


def _validate_framework_build_binding(
    candidate: Path, build: Mapping[str, Any], critical_files: Mapping[str, Any],
) -> None:
    # A host archive can already contain the right framework. An optional
    # override records its source; it must never enable or disable pairing checks.
    try:
        runtime_files = derived_package._paired_runtime_files(candidate)
        required_version = derived_package._required_maafw_version(candidate)
    except (OSError, UnicodeError, derived_package.BuildError) as error:
        raise ReleaseBuildError(f"candidate framework runtime pairing failed: {error}") from error
    for relative, actual_hash in runtime_files.items():
        if (
            not isinstance(critical_files.get(relative), str)
            or critical_files[relative].casefold() != actual_hash.casefold()
        ):
            raise ReleaseBuildError("candidate framework runtime is not bound to the critical inventory")
    if not isinstance(build.get("python_packages"), dict) or build["python_packages"].get("maafw") != required_version:
        raise ReleaseBuildError("candidate framework runtime version inventory is not paired")
    framework = build.get("framework")
    if framework is None:
        return
    if not isinstance(framework, dict) or set(framework) != {
        "repository", "version", "archive", "sha256", "files", "license",
    }:
        raise ReleaseBuildError("candidate framework provenance is invalid")
    version = framework.get("version")
    if (
        framework.get("repository") != "https://github.com/MaaXYZ/MaaFramework"
        or not isinstance(version, str)
        or re.fullmatch(r"\d+\.\d+\.\d+", version) is None
        or framework.get("archive") != f"MAA-win-x86_64-v{version}.zip"
        or not isinstance(framework.get("sha256"), str)
        or SHA256_PATTERN.fullmatch(framework["sha256"]) is None
    ):
        raise ReleaseBuildError("candidate framework archive identity is invalid")
    files = framework.get("files")
    native_root = "runtimes/win-x64/native"
    python_native_root = "python/Lib/site-packages/maa/bin"
    metadata_relative = f"python/Lib/site-packages/maafw-{version}.dist-info/METADATA"
    notice_relative = "THIRD_PARTY_NOTICES/MaaFramework-LICENSE"
    roots = (
        native_root + "/", python_native_root + "/",
        "MaaAgentBinary/", "libs/MaaAgentBinary/", "share/MaaAgentBinary/",
    )
    required = {"requirements.txt", metadata_relative, notice_relative}
    for name in ("MaaFramework.dll", "MaaAgentClient.dll", "MaaAgentServer.dll"):
        required.update((f"{native_root}/{name}", f"{python_native_root}/{name}"))
    if not isinstance(files, dict) or not required.issubset(files):
        raise ReleaseBuildError("candidate framework paired file inventory is incomplete")
    for relative, expected_hash in files.items():
        if (
            not isinstance(relative, str)
            or "\\" in relative
            or (relative not in required and not relative.startswith(roots))
            or not isinstance(expected_hash, str)
            or SHA256_PATTERN.fullmatch(expected_hash) is None
            or not isinstance(critical_files.get(relative), str)
            or critical_files[relative].casefold() != expected_hash.casefold()
        ):
            raise ReleaseBuildError("candidate framework file is not bound to the critical inventory")
        path = _resolve_inside(candidate, relative)
        if not path.is_file() or sha256_file(path).casefold() != expected_hash.casefold():
            raise ReleaseBuildError(f"candidate framework payload hash mismatch: {relative}")
    for root_text in (native_root, python_native_root):
        root = candidate / root_text
        for path in root.rglob("*"):
            if path.is_file() and path.relative_to(candidate).as_posix() not in files:
                raise ReleaseBuildError("candidate framework inventory omits a native payload")
    for relative in files:
        if relative.startswith(python_native_root + "/"):
            host_relative = native_root + relative[len(python_native_root):]
            if host_relative in files and files[host_relative].casefold() != files[relative].casefold():
                raise ReleaseBuildError("candidate framework host/Python DLL versions differ")
    metadata_files = tuple(
        path for path in (candidate / "python/Lib/site-packages").glob("*.dist-info/METADATA")
        if path.parent.name.casefold().startswith("maafw-")
    )
    if len(metadata_files) != 1 or metadata_files[0] != candidate / metadata_relative:
        raise ReleaseBuildError("candidate framework maafw metadata is missing or ambiguous")
    package = Parser().parsestr(metadata_files[0].read_text(encoding="utf-8"))
    requirements = (candidate / "requirements.txt").read_text(encoding="utf-8-sig").splitlines()
    maafw_lines = [
        line.partition("#")[0].strip() for line in requirements
        if re.match(r"\s*maafw(?:\s|[=<>!~;\[]|$)", line, flags=re.IGNORECASE)
    ]
    if (
        package.get("Name", "").casefold() != "maafw"
        or package.get("Version") != version
        or len(maafw_lines) != 1
        or re.fullmatch(r"maafw\s*==\s*" + re.escape(version), maafw_lines[0], flags=re.IGNORECASE) is None
        or not isinstance(build.get("python_packages"), dict)
        or build["python_packages"].get("maafw") != version
    ):
        raise ReleaseBuildError("candidate framework, Python package and maafw pin are not paired")
    license_record = framework.get("license")
    if license_record != {
        "spdx": "LGPL-3.0",
        "upstream_file": "LICENSE.md",
        "install_path": notice_relative,
        "sha256": files[notice_relative],
    }:
        raise ReleaseBuildError("candidate framework license provenance is invalid")


def _validate_candidate(
    candidate: Path,
    expected_version: str,
    expected_repository: str,
    *,
    expected_channel: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    forbidden = [relative for relative in FORBIDDEN_MUTABLE_PATHS if (candidate / relative).exists()]
    if forbidden:
        raise ReleaseBuildError(f"candidate contains mutable runtime state: {', '.join(forbidden)}")
    build = _load_object(candidate / "GAKUMAS_HELPER_BUILD.json")
    interface = _load_object(candidate / "interface.json")
    if build.get("schema_version") != 1 or build.get("product") != "MaaGakumasu":
        raise ReleaseBuildError("candidate build manifest identity is invalid")
    if build.get("derived_version") != expected_version or interface.get("version") != expected_version:
        raise ReleaseBuildError("candidate version does not match the release tag")
    update_contract = build.get("update_contract")
    if not isinstance(update_contract, dict) or update_contract.get("mode") != "derived_release_channel":
        raise ReleaseBuildError("candidate is not configured for the durable derived release channel")
    if update_contract.get("repository") != expected_repository or interface.get("github") != expected_repository:
        raise ReleaseBuildError("candidate update repository does not match the release repository")
    forbidden_update_keys = sorted(FORBIDDEN_DERIVED_UPDATE_KEYS & interface.keys())
    if forbidden_update_keys:
        raise ReleaseBuildError(
            "candidate interface retains forbidden upstream update routes: "
            + ", ".join(forbidden_update_keys)
        )
    expected_update_contract = {
        "version_namespace": "independent_gkh_semver",
        "client_updater": "mfa_builtin_resource_update",
        "payload_scope": "full_derived_package",
        "python_dependency_updater": "existing_agent_pip_update",
    }
    if any(update_contract.get(key) != value for key, value in expected_update_contract.items()):
        raise ReleaseBuildError(
            "candidate update contract must use the MFA built-in full-package update path"
        )
    if update_contract.get("release_channel") != expected_channel:
        raise ReleaseBuildError(
            f"candidate must record the {expected_channel} release channel for its release qualification"
        )
    previous_channel_version = update_contract.get("previous_channel_version")
    if expected_version == FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION:
        if previous_channel_version != LAST_PUBLISHED_LEGACY_CHANNEL_VERSION:
            raise ReleaseBuildError(
                "first publishable independent GKH release must migrate from the last published "
                f"legacy release {LAST_PUBLISHED_LEGACY_CHANNEL_VERSION}"
            )
        if (
            update_contract.get("version_ordering")
            != "legacy_combined_to_independent_semver_manual_bootstrap"
            or update_contract.get("requires_manual_bootstrap") is not True
        ):
            raise ReleaseBuildError(
                "first publishable independent GKH release must preserve the one-time legacy "
                "manual migration contract"
            )
    else:
        if (
            not isinstance(previous_channel_version, str)
            or ARENA_PREVIEW_VERSION_PATTERN.fullmatch(previous_channel_version) is None
            or previous_channel_version in RESERVED_UNPUBLISHABLE_PROJECT_VERSIONS
            or _project_version_key(previous_channel_version)
            < _project_version_key(FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION)
            or _project_version_key(previous_channel_version)
            >= _project_version_key(expected_version)
            or update_contract.get("version_ordering")
            != "independent_gkh_semver_precedence"
            or update_contract.get("requires_manual_bootstrap") is not False
        ):
            raise ReleaseBuildError(
                "later independent GKH releases must advance an independent non-bootstrap channel version"
            )
    source = build.get("source")
    if (
        not isinstance(source, dict)
        or source.get("repository") != expected_repository
        or not isinstance(source.get("revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", source["revision"]) is None
    ):
        raise ReleaseBuildError("candidate public source provenance is invalid")

    critical_files = build.get("critical_files")
    if not isinstance(critical_files, dict) or not critical_files:
        raise ReleaseBuildError("candidate build manifest has no critical file inventory")
    _validate_mfa_core_build_binding(candidate, build, critical_files)
    _validate_framework_build_binding(candidate, build, critical_files)
    for relative, expected_hash in critical_files.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ReleaseBuildError("candidate critical file inventory is invalid")
        path = _resolve_inside(candidate, relative)
        if not path.is_file():
            raise ReleaseBuildError(f"candidate critical file is missing: {relative}")
        if sha256_file(path).casefold() != expected_hash.casefold():
            raise ReleaseBuildError(f"candidate critical file hash mismatch: {relative}")
    return build, interface


def _project_version_key(version: str) -> tuple[int, int, int]:
    match = ARENA_PREVIEW_VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ReleaseBuildError("release version must use independent GKH SemVer")
    return tuple(int(part) for part in version.removeprefix("v").split("."))


def _validate_packaged_p_item_qualification(candidate: Path, manifest: Mapping[str, Any]) -> None:
    """Bind derived calibration to the recognizer bytes actually being packaged.

    The fixed-tool check in the static promoter cannot stand in for this check:
    the inspected candidate may be a different directory. Never import its code.
    """
    source = manifest.get("source")
    provenance = source.get("extension_provenance") if isinstance(source, Mapping) else None
    if not isinstance(provenance, Mapping) or provenance.get("schema_version") != 5:
        return
    declaration = provenance.get(static_handoff.derived_source.SOURCE_KEY)
    qualification = declaration.get("qualification") if isinstance(declaration, Mapping) else None
    expected = qualification.get("reference_code_canonical_lf_sha256") if isinstance(qualification, Mapping) else None
    if not static_handoff._is_sha256(expected):
        raise ReleaseBuildError("derived P-item qualification lacks the packaged recognizer source binding")
    path = _resolve_inside(candidate, "agent/p_item_recognition/reference.py")
    if not path.is_file():
        raise ReleaseBuildError("derived P-item qualification requires packaged agent/p_item_recognition/reference.py")
    if static_handoff.derived_source.canonical_source_sha256(path) != str(expected).upper():
        raise ReleaseBuildError("packaged P-item recognizer source differs from the scoped qualification")


def _component_inventory(candidate: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    raw: dict[str, dict[str, Any]] = {}
    inventory: dict[str, Any] = {}
    for name, relative in COMPONENT_MANIFESTS.items():
        path = _resolve_inside(candidate, relative)
        manifest = _load_object(path)
        _validate_component_production_handoff(name, manifest)
        if name == "arena_engine_catalog":
            _validate_ris_engine_production_handoff(manifest)
        raw[name] = manifest
        inventory[name] = {
            "manifest": relative,
            "manifest_sha256": sha256_file(path),
        }

    card = raw["card_vision"]
    arena_card = raw["arena_card_vision"]
    p_item = raw["p_item_reference"]
    badge = raw["arena_badge_reference"]
    cost = raw["arena_cost_reference"]
    engine = raw["arena_engine_catalog"]
    try:
        if engine["adapter_protocol_version"] != "2.0" or engine["calibration_schema_version"] != 8:
            raise ReleaseBuildError("arena engine protocol or calibration schema is incompatible")

        declared_files = (
            ("card_vision", card["model"]["path"], card["model"]["sha256"]),
            ("card_vision", card["gallery"]["path"], card["gallery"]["sha256"]),
            ("card_vision", card["gallery"]["class_table_path"], card["gallery"]["class_table_sha256"]),
            ("arena_card_vision", arena_card["model"]["path"], arena_card["model"]["sha256"]),
            ("arena_card_vision", arena_card["gallery"]["path"], arena_card["gallery"]["sha256"]),
            (
                "arena_card_vision",
                arena_card["gallery"]["class_table_path"],
                arena_card["gallery"]["class_table_sha256"],
            ),
            ("arena_badge_reference", badge["gallery"]["path"], badge["gallery"]["sha256"]),
            ("arena_cost_reference", cost["gallery"]["path"], cost["gallery"]["sha256"]),
            ("p_item_reference", p_item["reference_gallery"]["path"], p_item["reference_gallery"]["sha256"]),
        )
    except (KeyError, TypeError) as error:
        raise ReleaseBuildError("component compatibility metadata is incomplete") from error

    for component, relative, expected_hash in declared_files:
        manifest_root = _resolve_inside(candidate, COMPONENT_MANIFESTS[component]).parent
        asset = _resolve_inside(manifest_root, str(relative))
        if not asset.is_file():
            raise ReleaseBuildError(f"component asset is missing: {component}/{relative}")
        if sha256_file(asset).casefold() != str(expected_hash).casefold():
            raise ReleaseBuildError(f"component asset hash mismatch: {component}/{relative}")

    engine_root = _resolve_inside(candidate, COMPONENT_MANIFESTS["arena_engine_catalog"]).parent
    cost_catalog_dir = _resolve_inside(
        engine_root,
        SKILL_CARD_CATALOG_RELATIVE_PATH,
    ).parent
    for component in ("arena_cost_reference", "p_item_reference"):
        manifest_root = _resolve_inside(
            candidate,
            COMPONENT_MANIFESTS[component],
        ).parent
        try:
            validate_promoted_component(
                component,
                manifest_root,
                raw[component],
                cost_catalog_dir=(
                    cost_catalog_dir
                    if component == "arena_cost_reference"
                    else None
                ),
                p_item_production_source_evidence_path=(
                    _select_p_item_production_source_evidence(
                        manifest_root, raw[component]
                    )
                    if component == "p_item_reference"
                    else None
                ),
            )
            if component == "p_item_reference":
                _validate_packaged_p_item_qualification(candidate, raw[component])
        except StaticReferenceHandoffError as error:
            raise ReleaseBuildError(
                f"{component} production handoff validation failed: {error}"
            ) from error

    card_root = _resolve_inside(candidate, COMPONENT_MANIFESTS["card_vision"]).parent
    arena_card_root = _resolve_inside(
        candidate,
        COMPONENT_MANIFESTS["arena_card_vision"],
    ).parent
    badge_root = _resolve_inside(
        candidate,
        COMPONENT_MANIFESTS["arena_badge_reference"],
    ).parent
    card_class_names, card_rows, card_ids, card_groups, card_group_by_id = (
        _load_skill_card_gallery_contract(
        _resolve_inside(card_root, str(card["gallery"]["path"])),
        label="card vision",
        )
    )
    arena_class_names, arena_rows, arena_ids, arena_groups, arena_group_by_id = (
        _load_skill_card_gallery_contract(
        _resolve_inside(arena_card_root, str(arena_card["gallery"]["path"])),
        label="arena card vision",
        )
    )
    arena_reference_rows_by_id: dict[int, int] = {}
    for _, raw_business_id, _ in arena_rows:
        business_id = int(raw_business_id)
        arena_reference_rows_by_id[business_id] = (
            arena_reference_rows_by_id.get(business_id, 0) + 1
        )
    (
        badge_ids,
        badge_groups,
        badge_group_by_id,
        badge_reference_row_count,
        badge_reference_rows_by_id,
        badge_live_reference_rows,
    ) = _load_badge_gallery_contract(
        _resolve_inside(badge_root, str(badge["gallery"]["path"])),
    )
    _validate_badge_live_evaluation(
        badge_root,
        badge,
        arena_card_manifest=arena_card,
        arena_reference_rows_by_business_id=arena_reference_rows_by_id,
        actual_reference_row_count=badge_reference_row_count,
        actual_live_reference_row_count=len(badge_live_reference_rows),
        actual_business_id_count=len(badge_ids),
        actual_visual_group_count=len(badge_groups),
        actual_group_by_business_id=badge_group_by_id,
        actual_reference_rows_by_business_id=badge_reference_rows_by_id,
        actual_live_reference_rows=badge_live_reference_rows,
    )
    arena_card_manifest_path = _resolve_inside(
        candidate,
        COMPONENT_MANIFESTS["arena_card_vision"],
    )
    arena_card_gallery_path = _resolve_inside(
        arena_card_manifest_path.parent,
        str(arena_card["gallery"]["path"]),
    )
    badge_source = badge.get("source")
    if (
        not isinstance(badge_source, Mapping)
        or not isinstance(
            badge_source.get("arena_component_manifest_sha256"),
            str,
        )
        or badge_source["arena_component_manifest_sha256"].casefold()
        != sha256_file(arena_card_manifest_path).casefold()
        or not isinstance(badge_source.get("arena_gallery_sha256"), str)
        or badge_source["arena_gallery_sha256"].casefold()
        != sha256_file(arena_card_gallery_path).casefold()
    ):
        raise ReleaseBuildError(
            "arena badge reference is not bound to the exact arena card "
            "manifest and gallery"
        )
    card_classes = tuple(
        str(value)
        for value in _load_array(
            _resolve_inside(card_root, str(card["gallery"]["class_table_path"]))
        )
    )
    arena_classes = tuple(
        str(value)
        for value in _load_array(
            _resolve_inside(
                arena_card_root,
                str(arena_card["gallery"]["class_table_path"]),
            )
        )
    )
    if card_classes != card_class_names or arena_classes != arena_class_names:
        raise ReleaseBuildError("skill-card class table and gallery order disagree")
    if card_ids != arena_ids or card_ids != badge_ids:
        raise ReleaseBuildError("skill-card recognition business-ID sets disagree")
    if card_groups != arena_groups or card_groups != badge_groups:
        raise ReleaseBuildError("skill-card recognition visual-group sets disagree")
    if card_rows != arena_rows:
        raise ReleaseBuildError(
            "skill-card base and arena gallery identity rows disagree"
        )
    if (
        card_group_by_id != arena_group_by_id
        or card_group_by_id != badge_group_by_id
    ):
        raise ReleaseBuildError(
            "skill-card recognition business-ID to visual-group mappings disagree"
        )
    _validate_skill_card_source_summary(
        card.get("source"),
        actual_ids=card_ids,
        actual_groups=card_groups,
        label="card vision",
    )
    _validate_skill_card_source_summary(
        arena_card.get("source"),
        actual_ids=arena_ids,
        actual_groups=arena_groups,
        label="arena card vision",
    )
    badge_business_count = badge["gallery"].get("business_card_id_count")
    badge_group_count = badge["gallery"].get("visual_group_count")
    if (
        isinstance(badge_business_count, bool)
        or not isinstance(badge_business_count, int)
        or badge_business_count != len(badge_ids)
        or isinstance(badge_group_count, bool)
        or not isinstance(badge_group_count, int)
        or badge_group_count != len(badge_groups)
    ):
        raise ReleaseBuildError("arena badge manifest counts disagree with the gallery")

    engine_files = engine.get("files")
    if not isinstance(engine_files, dict) or not engine_files:
        raise ReleaseBuildError("arena engine manifest has no file inventory")
    for relative, expected_hash in engine_files.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ReleaseBuildError("arena engine file inventory is invalid")
        asset = _resolve_inside(engine_root, relative)
        if not asset.is_file():
            raise ReleaseBuildError(f"arena engine asset is missing: {relative}")
        if sha256_file(asset).casefold() != expected_hash.casefold():
            raise ReleaseBuildError(f"arena engine asset hash mismatch: {relative}")
    actual_engine_files = {
        path.relative_to(engine_root).as_posix() for path in engine_root.rglob("*") if path.is_file()
    }
    expected_engine_files = {
        "manifest.json",
        *(str(relative).replace("\\", "/") for relative in engine_files),
    }
    if actual_engine_files != expected_engine_files:
        extra = sorted(actual_engine_files - expected_engine_files)
        missing = sorted(expected_engine_files - actual_engine_files)
        raise ReleaseBuildError(f"arena engine file inventory mismatch: extra={extra}, missing={missing}")
    _validate_ris_engine_stage_binding(engine_root, engine)

    catalog_ids = _load_skill_card_catalog_ids(
        _resolve_inside(engine_root, SKILL_CARD_CATALOG_RELATIVE_PATH)
    )
    missing_catalog_ids = sorted(card_ids - catalog_ids)
    if missing_catalog_ids:
        raise ReleaseBuildError(
            f"skill-card recognition IDs are absent from the arena engine catalog: {missing_catalog_ids}"
        )
    missing_recognition_ids = sorted(catalog_ids - card_ids)
    if missing_recognition_ids:
        raise ReleaseBuildError(
            "arena engine catalog skill-card IDs are absent from the recognition assets: "
            f"{missing_recognition_ids}"
        )

    p_item_root = _resolve_inside(candidate, COMPONENT_MANIFESTS["p_item_reference"]).parent
    p_item_gallery_path = _resolve_inside(
        p_item_root,
        str(p_item["reference_gallery"]["path"]),
    )
    _validate_p_item_release_contract(
        p_item,
        gallery_path=p_item_gallery_path,
        catalog_path=_resolve_inside(engine_root, P_ITEM_CATALOG_RELATIVE_PATH),
    )
    return inventory, raw


def _write_deterministic_zip(candidate: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for source in sorted((path for path in candidate.rglob("*") if path.is_file()), key=lambda path: path.relative_to(candidate).as_posix()):
            relative = source.relative_to(candidate).as_posix()
            info = zipfile.ZipInfo(relative, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (source.stat().st_mode & 0xFFFF) << 16
            with source.open("rb") as input_stream, archive.open(info, "w", force_zip64=True) as output_stream:
                for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                    output_stream.write(chunk)


def build_release_assets(
    *,
    candidate: Path,
    output_dir: Path,
    release_version: str,
    release_repository: str,
    qualification: str = "arena_preview",
) -> dict[str, Path]:
    candidate = candidate.resolve()
    output_dir = output_dir.resolve()
    if qualification not in RELEASE_CHANNELS:
        raise ReleaseBuildError("release qualification must be arena_preview or arena_release")
    if not candidate.is_dir():
        raise ReleaseBuildError(f"candidate directory is missing: {candidate}")
    version_match = ARENA_PREVIEW_VERSION_PATTERN.fullmatch(release_version)
    if version_match is None:
        raise ReleaseBuildError(
            "arena releases must use independent GKH SemVer vMAJOR.MINOR.PATCH"
        )
    if _project_version_key(release_version) < _project_version_key(
        MINIMUM_INDEPENDENT_PROJECT_VERSION
    ):
        raise ReleaseBuildError(
            "arena release version must not precede the first independent GKH version "
            f"{MINIMUM_INDEPENDENT_PROJECT_VERSION}"
        )
    if release_version in RESERVED_UNPUBLISHABLE_PROJECT_VERSIONS:
        raise ReleaseBuildError(
            f"arena release version {release_version} is permanently reserved after a failed "
            "pre-publication install and must not be rebuilt or published; use at least "
            f"{FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION}"
        )

    build, interface = _validate_candidate(
        candidate,
        release_version,
        release_repository,
        expected_channel=RELEASE_CHANNELS[qualification],
    )
    upstream = build.get("upstream")
    if not isinstance(upstream, dict) or re.fullmatch(
        r"v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)",
        str(upstream.get("tag", "")),
    ) is None:
        raise ReleaseBuildError(
            "candidate must record its upstream Maa version separately from the GKH version"
        )
    components, raw = _component_inventory(candidate)
    _validate_ris_engine_build_binding(
        build,
        raw["arena_engine_catalog"],
        _resolve_inside(candidate, COMPONENT_MANIFESTS["arena_engine_catalog"]),
    )
    required_notices = (
        candidate / "LICENSE",
        candidate / "ASSET_PROVENANCE.md",
        candidate / "THIRD_PARTY_NOTICES" / "gakumas-tools-LICENSE",
        candidate / "THIRD_PARTY_NOTICES" / "GAME-CONTENT-NOTICE.md",
        candidate / MFA_CORE_NOTICE_PATH,
        candidate / "assets" / "arena-winrate" / "THIRD_PARTY_NOTICES" / "gakumas-tools-LICENSE",
        candidate / "assets" / "arena-winrate" / "THIRD_PARTY_NOTICES" / "Node.js-LICENSE",
    )
    missing_notices = [str(path.relative_to(candidate)) for path in required_notices if not path.is_file()]
    if missing_notices:
        raise ReleaseBuildError(f"candidate is missing release notices: {', '.join(missing_notices)}")
    privacy = _validate_release_privacy(candidate)

    output_dir.mkdir(parents=True, exist_ok=True)
    package_name = f"MaaGakumasu-{PLATFORM}-{release_version}.zip"
    manifest_name = f"GakumasHelper-release-{release_version}.json"
    checksums_name = f"GakumasHelper-checksums-{release_version}.txt"
    final_paths = {name: output_dir / name for name in (package_name, manifest_name, checksums_name)}
    existing = [str(path) for path in final_paths.values() if path.exists()]
    if existing:
        raise ReleaseBuildError(f"release output already exists: {', '.join(existing)}")

    with tempfile.TemporaryDirectory(prefix="gakumas-release-", dir=output_dir) as temporary:
        temporary_root = Path(temporary)
        package_path = temporary_root / package_name
        _write_deterministic_zip(candidate, package_path)
        _validate_written_zip(
            candidate,
            package_path,
            expected_mfa_core_sha256=build["mfa_core"]["payload"]["sha256"],
        )
        package_sha256 = sha256_file(package_path)

        card = raw["card_vision"]
        arena_card = raw["arena_card_vision"]
        p_item = raw["p_item_reference"]
        badge = raw["arena_badge_reference"]
        engine = raw["arena_engine_catalog"]
        arena_card_root = _resolve_inside(
            candidate,
            COMPONENT_MANIFESTS["arena_card_vision"],
        ).parent
        _, _, release_arena_ids, _, _ = _load_skill_card_gallery_contract(
            _resolve_inside(arena_card_root, str(arena_card["gallery"]["path"])),
            label="arena card vision",
        )
        release_manifest = {
            "schema_version": SCHEMA_VERSION,
            "product": "MaaGakumasu-GakumasHelper",
            "qualification": qualification,
            "release": {
                "version": release_version,
                "channel": RELEASE_CHANNELS[qualification],
                "repository": release_repository,
                "asset": package_name,
                "asset_bytes": package_path.stat().st_size,
                "asset_sha256": package_sha256,
                "platform": PLATFORM,
            },
            "application": {
                "interface_version": interface.get("interface_version"),
                "source_repository": build["source"]["repository"],
                "source_revision": build["source"]["revision"],
                "upstream": build["upstream"],
                "mfa_core": build["mfa_core"],
                "python_packages": build.get("python_packages", {}),
                **({"framework": build["framework"]} if build.get("framework") is not None else {}),
            },
            "compatibility": {
                "card_business_id_count": card["source"]["business_card_id_count"],
                "card_visual_identity_count": card["source"]["visual_identity_count"],
                "card_embedding_dimension": card["gallery"]["embedding_dim"],
                "arena_card_business_id_max": max(release_arena_ids),
                "arena_badge_rule_version": badge["runtime"]["reference_rule_version"],
                "p_item_business_id_count": p_item["reference_gallery"]["business_id_count"],
                "p_item_minimum_capture_size": p_item["runtime"]["coarse_fine_profiles"][0]["minimum_capture_size"],
                "arena_engine_revision": engine["commit"],
                "arena_adapter_protocol": engine["adapter_protocol_version"],
                "arena_calibration_schema": engine["calibration_schema_version"],
            },
            "components": components,
            "redistribution": {
                "project_and_upstream_license": "AGPL-3.0",
                "gakumas_tools_license": engine.get("license"),
                "self_trained_assets": "approved_for_this_release_by_human_gate",
                "user_runtime_capture_or_private_training_corpus_in_release": False,
                "upstream_resource_templates_in_release": True,
                "notices_embedded": [str(path.relative_to(candidate)).replace("\\", "/") for path in required_notices],
                "notice_sha256": {
                    str(path.relative_to(candidate)).replace("\\", "/"): sha256_file(path)
                    for path in required_notices
                },
                "dependency_source_normalizations": build.get("python_source_normalizations", {}),
            },
            "privacy_gate": {
                "schema_version": 1,
                "policy": "reject_local_runtime_state_private_captures_machine_paths_and_credentials",
                "files_scanned": privacy["files_scanned"],
                "bytes_scanned": privacy["bytes_scanned"],
            },
            "known_gates": [
                "P-item upgraded-marker and broader real-window evidence remain open",
                "Arena reader second-window blind validation remains open",
                "Arena win-rate final product acceptance remains preview-only",
            ] if qualification == "arena_preview" else [
                "Result transition recovery has not yet passed new live-device validation",
                "A complete arena daily challenge at 540x960 has not yet passed",
                "Uninterrupted two-day arena daily acceptance remains open",
            ],
        }
        manifest_path = temporary_root / manifest_name
        manifest_payload = (json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        if INTERNAL_TASK_IDENTIFIER_PATTERN.search(manifest_payload):
            raise ReleaseBuildError("release manifest contains an internal task identifier")
        manifest_path.write_bytes(manifest_payload)
        manifest_sha256 = sha256_file(manifest_path)
        checksums_path = temporary_root / checksums_name
        checksums_path.write_text(
            f"{package_sha256}  {package_name}\n{manifest_sha256}  {manifest_name}\n",
            encoding="ascii",
        )

        for name, final_path in final_paths.items():
            os.replace(temporary_root / name, final_path)
    return {"package": final_paths[package_name], "manifest": final_paths[manifest_name], "checksums": final_paths[checksums_name]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--release-repository", required=True)
    parser.add_argument("--qualification", choices=tuple(RELEASE_CHANNELS), default="arena_preview")
    return parser


def main() -> int:
    args = _parser().parse_args()
    assets = build_release_assets(
        candidate=args.candidate,
        output_dir=args.output_dir,
        release_version=args.release_version,
        release_repository=args.release_repository,
        qualification=args.qualification,
    )
    print(json.dumps({name: str(path) for name, path in assets.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
