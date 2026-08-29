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

try:
    from tools.deployment.privacy_gate import PrivacyGateError, validate_tree
except ModuleNotFoundError:  # Direct script execution from tools/deployment.
    from privacy_gate import PrivacyGateError, validate_tree

SCHEMA_VERSION = 1
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


def _validate_written_zip(candidate: Path, package: Path) -> None:
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
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ReleaseBuildError(f"release ZIP integrity check failed: {corrupt}")


def _validate_candidate(candidate: Path, expected_version: str, expected_repository: str) -> tuple[dict[str, Any], dict[str, Any]]:
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
    if update_contract.get("release_channel") != "beta":
        raise ReleaseBuildError("arena preview candidate must record the beta release channel")
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


def _component_inventory(candidate: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    raw: dict[str, dict[str, Any]] = {}
    inventory: dict[str, Any] = {}
    for name, relative in COMPONENT_MANIFESTS.items():
        path = _resolve_inside(candidate, relative)
        manifest = _load_object(path)
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
        if card["source"]["business_card_id_count"] != 856 or card["source"]["visual_identity_count"] != 439:
            raise ReleaseBuildError("card vision catalog cardinality is incompatible")
        if arena_card["source"]["business_card_id_max"] != 856 or arena_card["source"]["visual_identity_count"] != 439:
            raise ReleaseBuildError("arena card catalog cardinality is incompatible")
        if badge["gallery"]["business_card_id_count"] != 856 or badge["gallery"]["visual_group_count"] != 439:
            raise ReleaseBuildError("arena badge reference catalog is incompatible")
        if p_item["reference_gallery"]["business_id_count"] != 420:
            raise ReleaseBuildError("P-item reference catalog is incompatible")
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
    if qualification != "arena_preview":
        raise ReleaseBuildError("this release path is currently approved only for the arena preview qualification")
    if not candidate.is_dir():
        raise ReleaseBuildError(f"candidate directory is missing: {candidate}")
    version_match = ARENA_PREVIEW_VERSION_PATTERN.fullmatch(release_version)
    if version_match is None:
        raise ReleaseBuildError(
            "arena preview releases must use independent GKH SemVer vMAJOR.MINOR.PATCH"
        )
    if _project_version_key(release_version) < _project_version_key(
        MINIMUM_INDEPENDENT_PROJECT_VERSION
    ):
        raise ReleaseBuildError(
            "arena preview version must not precede the first independent GKH version "
            f"{MINIMUM_INDEPENDENT_PROJECT_VERSION}"
        )
    if release_version in RESERVED_UNPUBLISHABLE_PROJECT_VERSIONS:
        raise ReleaseBuildError(
            f"arena preview version {release_version} is permanently reserved after a failed "
            "pre-publication install and must not be rebuilt or published; use at least "
            f"{FIRST_PUBLISHABLE_INDEPENDENT_PROJECT_VERSION}"
        )

    build, interface = _validate_candidate(candidate, release_version, release_repository)
    upstream = build.get("upstream")
    if not isinstance(upstream, dict) or re.fullmatch(
        r"v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)",
        str(upstream.get("tag", "")),
    ) is None:
        raise ReleaseBuildError(
            "candidate must record its upstream Maa version separately from the GKH version"
        )
    components, raw = _component_inventory(candidate)
    required_notices = (
        candidate / "LICENSE",
        candidate / "ASSET_PROVENANCE.md",
        candidate / "THIRD_PARTY_NOTICES" / "gakumas-tools-LICENSE",
        candidate / "THIRD_PARTY_NOTICES" / "GAME-CONTENT-NOTICE.md",
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
        _validate_written_zip(candidate, package_path)
        package_sha256 = sha256_file(package_path)

        card = raw["card_vision"]
        arena_card = raw["arena_card_vision"]
        p_item = raw["p_item_reference"]
        badge = raw["arena_badge_reference"]
        engine = raw["arena_engine_catalog"]
        release_manifest = {
            "schema_version": SCHEMA_VERSION,
            "product": "MaaGakumasu-GakumasHelper",
            "qualification": qualification,
            "release": {
                "version": release_version,
                "channel": "beta",
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
                "python_packages": build.get("python_packages", {}),
            },
            "compatibility": {
                "card_business_id_count": card["source"]["business_card_id_count"],
                "card_visual_identity_count": card["source"]["visual_identity_count"],
                "card_embedding_dimension": card["gallery"]["embedding_dim"],
                "arena_card_business_id_max": arena_card["source"]["business_card_id_max"],
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
    parser.add_argument("--qualification", default="arena_preview")
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
