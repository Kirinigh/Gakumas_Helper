"""Materialize one offline, append-only skill-card dataset extension.

The command never accesses the network.  It joins already-frozen catalog,
decoded-master, official-manifest, raw-art, and fixed-icon inputs, then emits the
small EXTENSION dataset consumed by ``build_card_embedding_gallery.py``.
Accepted historical rows are compared with the accepted dataset lineage so an
upstream rewrite cannot silently pass as a new-card batch.
"""

from __future__ import annotations

import io
import os
import re
import csv
import sys
import json
import uuid
import zlib
import shutil
import hashlib
import argparse
import unicodedata
from typing import Any, Iterable, Sequence
from pathlib import Path
from dataclasses import dataclass

import PIL
from PIL import Image

SOURCE_DOMAIN = "OFFICIAL_RAW_CARD_ART"
CATALOG_REPOSITORY = "https://github.com/surisuririsu/gakumas-tools"
CATALOG_LICENSE = "BSD-3-Clause"
NAME_ALIASES = {
    "ストレッチ談義": "ストレッチ談議",
    "愛をこめて": "愛を込めて",
    "月明りに包まれて": "月明かりに包まれて",
    "夢と現境界線": "夢と現の境界線",
}
IDENTITY_FIELDS = (
    "catalog_name",
    "api_name",
    "internal_id",
    "upgrade_count",
    "asset_id",
    "is_character_asset",
    "asset_variants",
)
SAFE_ASSET_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ExtensionDatasetError(RuntimeError):
    """Raised when the frozen update batch is not a safe append-only extension."""


@dataclass(frozen=True)
class CatalogCard:
    business_id: int
    name: str
    upgraded: bool


@dataclass(frozen=True)
class CardMapping:
    business_id: int
    catalog_name: str
    api_name: str
    internal_id: str
    upgrade_count: int
    asset_id: str
    is_character_asset: bool
    asset_variants: tuple[str, ...]

    def crosswalk_row(self) -> dict[str, Any]:
        return {
            "business_id": self.business_id,
            "catalog_name": self.catalog_name,
            "api_name": self.api_name,
            "internal_id": self.internal_id,
            "upgrade_count": self.upgrade_count,
            "asset_id": self.asset_id,
            "is_character_asset": self.is_character_asset,
            "asset_variants": list(self.asset_variants),
        }


@dataclass(frozen=True)
class ExtensionDatasetResult:
    output_dir: Path
    manifest: dict[str, Any]
    extension_business_ids: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ExtensionDatasetError(f"source file is unavailable: {path}") from error
    return digest.hexdigest().upper()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(_json_bytes(payload))


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExtensionDatasetError(f"{label} is unavailable or invalid: {path}") from error


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    value = _load_json(path, label=label)
    if not isinstance(value, dict):
        raise ExtensionDatasetError(f"{label} must be a JSON object")
    return value


def _resolve_inside(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ExtensionDatasetError(f"{label} path is missing")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ExtensionDatasetError(f"{label} path must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ExtensionDatasetError(f"{label} path escapes its frozen root")
    return resolved


def _declared_sha256(value: object, *, label: str) -> str:
    text = str(value).upper()
    if len(text) != 64 or any(character not in "0123456789ABCDEF" for character in text):
        raise ExtensionDatasetError(f"{label} has an invalid SHA-256 declaration")
    return text


def _verify_file(path: Path, expected: object, *, label: str) -> str:
    declared = _declared_sha256(expected, label=label)
    actual = sha256_file(path)
    if actual != declared:
        raise ExtensionDatasetError(f"{label} SHA-256 mismatch: {path}")
    return actual


def _positive_id(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ExtensionDatasetError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ExtensionDatasetError(f"{label} must be a positive integer") from error
    if parsed < 1 or str(value).strip() != str(parsed):
        raise ExtensionDatasetError(f"{label} must be a canonical positive integer")
    return parsed


def _bool_value(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
        return value.strip().casefold() == "true"
    raise ExtensionDatasetError(f"{label} must be boolean")


def _normalized_name(name: str, *, upgraded: bool = False) -> str:
    value = unicodedata.normalize("NFKC", name).replace(" ", "")
    for source, destination in NAME_ALIASES.items():
        value = value.replace(source, destination)
    if upgraded and not value.endswith("+"):
        value += "+"
    return value


def _unwrap_rows(value: Any, *, label: str, keys: Sequence[str]) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        candidates = [value.get(key) for key in keys if key in value]
        if len(candidates) != 1:
            raise ExtensionDatasetError(f"{label} JSON wrapper is ambiguous")
        value = candidates[0]
    if not isinstance(value, list) or not value or not all(isinstance(row, dict) for row in value):
        raise ExtensionDatasetError(f"{label} must contain a non-empty object list")
    return value


def load_catalog(path: Path) -> tuple[CatalogCard, ...]:
    """Load the current gakumas-tools JSON catalog or its CSV equivalent."""

    if path.suffix.casefold() == ".json":
        rows = _unwrap_rows(
            _load_json(path, label="skill-card catalog"),
            label="skill-card catalog",
            keys=("skill_cards", "cards", "items", "data"),
        )
    else:
        try:
            text = path.read_text(encoding="utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(text)))
        except (OSError, UnicodeError, csv.Error) as error:
            raise ExtensionDatasetError(f"skill-card catalog is unavailable or invalid: {path}") from error
        if not rows:
            raise ExtensionDatasetError("skill-card catalog is empty")

    cards: list[CatalogCard] = []
    seen: set[int] = set()
    for row in rows:
        business_id = _positive_id(row.get("id"), label="catalog business ID")
        name = row.get("name")
        if business_id in seen or not isinstance(name, str) or not name.strip():
            raise ExtensionDatasetError("catalog IDs and names must be unique, non-empty identities")
        seen.add(business_id)
        cards.append(
            CatalogCard(
                business_id=business_id,
                name=name,
                upgraded=_bool_value(row.get("upgraded"), label=f"catalog {business_id} upgraded"),
            )
        )
    return tuple(sorted(cards, key=lambda card: card.business_id))


def _load_pcard(path: Path) -> list[dict[str, Any]]:
    return _unwrap_rows(
        _load_json(path, label="decoded pcard master"),
        label="decoded pcard master",
        keys=("pcards", "cards", "items", "data"),
    )


def _official_assets(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("assetBundleList")
    if not isinstance(rows, list) or not rows:
        raise ExtensionDatasetError("official manifest assetBundleList is missing")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ExtensionDatasetError("official manifest contains an invalid asset record")
        name = row["name"]
        if name in result:
            raise ExtensionDatasetError(f"official manifest contains duplicate asset name: {name}")
        result[name] = row
    return result


def build_crosswalk(
    catalog: Sequence[CatalogCard],
    pcard_rows: Sequence[dict[str, Any]],
    official_assets: dict[str, dict[str, Any]],
) -> tuple[CardMapping, ...]:
    if len(pcard_rows) != len(catalog):
        raise ExtensionDatasetError("catalog and decoded pcard record counts differ")

    api_index: dict[str, dict[str, Any]] = {}
    for row in pcard_rows:
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ExtensionDatasetError("decoded pcard master contains an invalid name")
        key = _normalized_name(name)
        if key in api_index:
            raise ExtensionDatasetError(f"decoded pcard master has duplicate normalized name: {key}")
        api_index[key] = row

    mappings: list[CardMapping] = []
    internal_keys: set[tuple[str, int]] = set()
    for catalog_card in catalog:
        key = _normalized_name(catalog_card.name, upgraded=catalog_card.upgraded)
        api = api_index.get(key)
        if api is None:
            raise ExtensionDatasetError(f"catalog ID {catalog_card.business_id} has no unique decoded pcard match")
        internal_id = api.get("id")
        asset_id = api.get("assetId")
        upgrade_count = api.get("upgradeCount")
        is_character_asset = api.get("isCharacterAsset")
        if (
            not isinstance(internal_id, str)
            or not internal_id
            or not isinstance(asset_id, str)
            or not asset_id
            or not SAFE_ASSET_NAME.fullmatch(asset_id)
            or isinstance(upgrade_count, bool)
            or upgrade_count not in (0, 1)
            or not isinstance(is_character_asset, bool)
        ):
            raise ExtensionDatasetError(f"decoded pcard identity fields are invalid for catalog ID {catalog_card.business_id}")
        if upgrade_count != int(catalog_card.upgraded):
            raise ExtensionDatasetError(f"catalog/pcard upgrade state disagrees for ID {catalog_card.business_id}")
        internal_key = (internal_id, upgrade_count)
        if internal_key in internal_keys:
            raise ExtensionDatasetError(f"multiple catalog IDs map to pcard identity {internal_key}")
        internal_keys.add(internal_key)

        variants = tuple(sorted(name for name in official_assets if name == asset_id or name.startswith(f"{asset_id}-")))
        if not variants:
            raise ExtensionDatasetError(f"official manifest has no asset for {asset_id}")
        if is_character_asset:
            if asset_id in variants:
                raise ExtensionDatasetError(f"character-specific asset unexpectedly has unsuffixed object: {asset_id}")
        elif variants != (asset_id,):
            raise ExtensionDatasetError(f"non-character asset has ambiguous official mapping: {asset_id}")
        mappings.append(
            CardMapping(
                business_id=catalog_card.business_id,
                catalog_name=catalog_card.name,
                api_name=str(api["name"]),
                internal_id=internal_id,
                upgrade_count=upgrade_count,
                asset_id=asset_id,
                is_character_asset=is_character_asset,
                asset_variants=variants,
            )
        )
    return tuple(mappings)


def _exact_business_ids(source: dict[str, Any]) -> tuple[int, ...]:
    raw_ids = source.get("business_card_ids")
    if raw_ids is not None:
        if not isinstance(raw_ids, list):
            raise ExtensionDatasetError("base gallery business_card_ids must be a list")
        ids = tuple(_positive_id(value, label="base gallery business ID") for value in raw_ids)
    else:
        minimum = source.get("business_card_id_min")
        maximum = source.get("business_card_id_max")
        if minimum is None or maximum is None:
            raise ExtensionDatasetError("base gallery does not declare an exact business-card ID set")
        minimum_id = _positive_id(minimum, label="base gallery minimum business ID")
        maximum_id = _positive_id(maximum, label="base gallery maximum business ID")
        if maximum_id < minimum_id:
            raise ExtensionDatasetError("base gallery business-card range is inverted")
        ids = tuple(range(minimum_id, maximum_id + 1))
    if not ids or tuple(sorted(set(ids))) != ids:
        raise ExtensionDatasetError("base gallery business-card ID set is not unique and sorted")
    count = source.get("business_card_id_count")
    if count is not None and count != len(ids):
        raise ExtensionDatasetError("base gallery business-card count disagrees with its exact set")
    return ids


def _expected_lineage_hashes(gallery_manifest: dict[str, Any]) -> tuple[str, ...]:
    source = gallery_manifest.get("source")
    if not isinstance(source, dict):
        raise ExtensionDatasetError("base gallery source metadata is missing")
    latest = _declared_sha256(
        source.get("dataset_manifest_sha256"),
        label="base gallery dataset manifest",
    )
    mode = source.get("dataset_manifest_mode")
    if mode == "EXTENSION":
        build = gallery_manifest.get("build")
        lineage = build.get("base_dataset_lineage_manifest_sha256s") if isinstance(build, dict) else None
        if not isinstance(lineage, list) or not lineage:
            raise ExtensionDatasetError("extension base gallery has no dataset lineage")
        return tuple(_declared_sha256(value, label="base dataset lineage manifest") for value in [*lineage, latest])
    return (latest,)


def _scope_ids(manifest: dict[str, Any]) -> tuple[int, ...]:
    scope = manifest.get("scope")
    if not isinstance(scope, dict):
        raise ExtensionDatasetError("base dataset scope is missing")
    raw = scope.get("business_ids", scope.get("business_card_ids"))
    if raw is None:
        minimum = scope.get("business_id_min", scope.get("business_card_id_min"))
        maximum = scope.get("business_id_max", scope.get("business_card_id_max"))
        if minimum is None or maximum is None:
            raise ExtensionDatasetError("base dataset must declare an exact business-ID set")
        minimum_id = _positive_id(minimum, label="base dataset minimum business ID")
        maximum_id = _positive_id(maximum, label="base dataset maximum business ID")
        if maximum_id < minimum_id:
            raise ExtensionDatasetError("base dataset business-ID range is inverted")
        values = tuple(range(minimum_id, maximum_id + 1))
    else:
        if not isinstance(raw, list) or not raw:
            raise ExtensionDatasetError("base dataset exact business IDs are invalid")
        values = tuple(_positive_id(value, label="base dataset business ID") for value in raw)
    if tuple(sorted(set(values))) != values:
        raise ExtensionDatasetError("base dataset business IDs are not unique and sorted")
    return values


def _load_base_crosswalk_lineage(
    gallery_manifest: dict[str, Any],
    manifest_paths: Sequence[Path],
) -> dict[int, dict[str, Any]]:
    expected_hashes = _expected_lineage_hashes(gallery_manifest)
    if len(manifest_paths) != len(expected_hashes):
        raise ExtensionDatasetError("base dataset manifest lineage count disagrees with the accepted gallery")
    result: dict[int, dict[str, Any]] = {}
    for index, (path, expected_hash) in enumerate(zip(manifest_paths, expected_hashes, strict=True)):
        if sha256_file(path) != expected_hash:
            raise ExtensionDatasetError(f"base dataset lineage SHA-256 mismatch at layer {index}")
        manifest = _load_object(path, label=f"base dataset manifest layer {index}")
        validation = manifest.get("validation")
        if not isinstance(validation, dict) or validation.get("status") != "CLEAR":
            raise ExtensionDatasetError(f"base dataset layer {index} validation is not CLEAR")
        mapping = manifest.get("mapping")
        if not isinstance(mapping, dict):
            raise ExtensionDatasetError(f"base dataset layer {index} mapping is missing")
        crosswalk_path = _resolve_inside(
            path.parent,
            mapping.get("crosswalk_path"),
            label=f"base crosswalk layer {index}",
        )
        _verify_file(
            crosswalk_path,
            mapping.get("crosswalk_sha256"),
            label=f"base crosswalk layer {index}",
        )
        rows = _load_json(crosswalk_path, label=f"base crosswalk layer {index}")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ExtensionDatasetError(f"base crosswalk layer {index} is invalid")
        layer_ids = _scope_ids(manifest)
        rows_by_id: dict[int, dict[str, Any]] = {}
        for row in rows:
            business_id = _positive_id(row.get("business_id"), label="base crosswalk business ID")
            if business_id in rows_by_id:
                raise ExtensionDatasetError("base crosswalk contains duplicate business IDs")
            missing = [field for field in IDENTITY_FIELDS if field not in row]
            if missing:
                raise ExtensionDatasetError(f"base crosswalk ID {business_id} lacks frozen identity fields: {missing}")
            rows_by_id[business_id] = row
        if tuple(sorted(rows_by_id)) != layer_ids:
            raise ExtensionDatasetError(f"base crosswalk layer {index} disagrees with its scope")
        overlap = set(result).intersection(rows_by_id)
        if overlap:
            raise ExtensionDatasetError(f"base dataset lineage is not append-only; repeated IDs={sorted(overlap)[:20]}")
        result.update(rows_by_id)
    return result


def _compare_base_identity(
    base_rows: dict[int, dict[str, Any]],
    current: dict[int, CardMapping],
) -> None:
    for business_id, baseline in base_rows.items():
        mapping = current.get(business_id)
        if mapping is None:
            raise ExtensionDatasetError(f"accepted business ID disappeared from catalog: {business_id}")
        candidate = mapping.crosswalk_row()
        changed = [field for field in IDENTITY_FIELDS if baseline.get(field) != candidate[field]]
        if changed:
            raise ExtensionDatasetError(f"accepted business ID {business_id} identity fields changed: {changed}")


def _class_name(mapping: CardMapping, variant: str) -> str:
    if variant == mapping.asset_id:
        return str(mapping.business_id)
    prefix = f"{mapping.asset_id}-"
    if not variant.startswith(prefix) or not variant.removeprefix(prefix):
        raise ExtensionDatasetError(f"invalid asset variant for ID {mapping.business_id}: {variant}")
    return f"{mapping.business_id}_{variant.removeprefix(prefix)}"


def _png_chunks(path: Path) -> tuple[bytes, ...]:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ExtensionDatasetError(f"fixed icon is not PNG: {path}")
    chunks: list[bytes] = []
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        if offset + 12 > len(data):
            raise ExtensionDatasetError(f"fixed icon PNG is truncated: {path}")
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        offset += 12 + length
        if offset > len(data):
            raise ExtensionDatasetError(f"fixed icon PNG is truncated: {path}")
        chunks.append(chunk_type)
        if chunk_type == b"IEND":
            break
    if offset != len(data) or not chunks or chunks[0] != b"IHDR" or chunks[-1] != b"IEND":
        raise ExtensionDatasetError(f"fixed icon PNG structure is invalid: {path}")
    return tuple(chunks)


def _verify_fixed_icon(path: Path) -> tuple[int, int, str]:
    chunks = _png_chunks(path)
    unexpected = sorted(set(chunks) - {b"IHDR", b"IDAT", b"IEND"})
    if unexpected:
        names = [value.decode("ascii", errors="replace") for value in unexpected]
        raise ExtensionDatasetError(f"fixed icon contains metadata/ancillary chunks {names}: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            info = dict(image.info)
    except (OSError, ValueError) as error:
        raise ExtensionDatasetError(f"fixed icon cannot be decoded: {path}") from error
    if width < 1 or height < 1 or mode not in {"RGB", "RGBA"} or info:
        raise ExtensionDatasetError(f"fixed icon decode/metadata contract failed: {path}")
    return width, height, mode


def _verify_raw_art(
    record: dict[str, Any],
    raw_art_root: Path,
    official_record: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    asset_name = record.get("asset_name")
    if not isinstance(asset_name, str) or not SAFE_ASSET_NAME.fullmatch(asset_name):
        raise ExtensionDatasetError("raw-art inventory contains an invalid asset_name")
    source = _resolve_inside(raw_art_root, record.get("path"), label=f"raw art {asset_name}")
    image_sha256 = _verify_file(source, record.get("sha256"), label=f"raw art {asset_name}")
    try:
        with Image.open(source) as image:
            image.verify()
        with Image.open(source) as image:
            width, height = image.size
            mode = image.mode
    except (OSError, ValueError) as error:
        raise ExtensionDatasetError(f"raw art cannot be decoded: {source}") from error
    if width < 1 or height < 1:
        raise ExtensionDatasetError(f"raw art has empty dimensions: {source}")
    for field, actual in (("width", width), ("height", height), ("mode", mode)):
        if record.get(field) is not None and record[field] != actual:
            raise ExtensionDatasetError(f"raw-art inventory {field} mismatch for {asset_name}")
    comparisons = (
        ("official_object_name", "objectName", str),
        ("official_bundle_size", "size", int),
        ("official_bundle_md5", "md5", lambda value: str(value).upper()),
    )
    for inventory_key, official_key, transform in comparisons:
        if inventory_key not in record or official_key not in official_record:
            raise ExtensionDatasetError(f"raw-art inventory cannot be tied to official manifest for {asset_name}")
        if transform(record[inventory_key]) != transform(official_record[official_key]):
            raise ExtensionDatasetError(f"raw-art/official manifest {inventory_key} mismatch for {asset_name}")
    return source, {
        "asset_name": asset_name,
        "path": "",
        "sha256": image_sha256,
        "width": width,
        "height": height,
        "mode": mode,
        "official_object_name": str(official_record["objectName"]),
        "official_bundle_size": int(official_record["size"]),
        "official_bundle_md5": str(official_record["md5"]).upper(),
    }


def _business_scope(ids: Sequence[int]) -> dict[str, Any]:
    ordered = list(ids)
    scope: dict[str, Any] = {"business_ids": ordered}
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        scope["business_id_min"] = ordered[0]
        scope["business_id_max"] = ordered[-1]
    return scope


def _copy_inputs(
    *,
    temporary: Path,
    mappings: Sequence[CardMapping],
    official_assets: dict[str, dict[str, Any]],
    raw_inventory: dict[str, Any],
    raw_art_root: Path,
    fixed_icon_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    images = raw_inventory.get("images")
    duplicates = raw_inventory.get("duplicates", [])
    if not isinstance(images, list) or not all(isinstance(row, dict) for row in images):
        raise ExtensionDatasetError("raw-art inventory images are invalid")
    if duplicates not in ([], None):
        raise ExtensionDatasetError("raw-art inventory contains unresolved duplicate groups")
    inventory_by_name: dict[str, dict[str, Any]] = {}
    for row in images:
        name = row.get("asset_name")
        if not isinstance(name, str) or name in inventory_by_name:
            raise ExtensionDatasetError("raw-art inventory contains duplicate/invalid asset names")
        inventory_by_name[name] = row

    required_assets = {variant for mapping in mappings for variant in mapping.asset_variants}
    if set(inventory_by_name) != required_assets:
        raise ExtensionDatasetError(
            "raw-art inventory must exactly cover extension assets; "
            f"missing={sorted(required_assets - set(inventory_by_name))}, "
            f"surplus={sorted(set(inventory_by_name) - required_assets)}"
        )

    art_output = temporary / "art"
    icon_output = temporary / "icons"
    art_output.mkdir()
    icon_output.mkdir()
    art_records: list[dict[str, Any]] = []
    art_hash_owners: dict[str, str] = {}
    for asset_name in sorted(required_assets):
        source, output_record = _verify_raw_art(
            inventory_by_name[asset_name],
            raw_art_root,
            official_assets[asset_name],
        )
        previous_asset = art_hash_owners.setdefault(output_record["sha256"], asset_name)
        if previous_asset != asset_name:
            raise ExtensionDatasetError(f"raw art is byte-identical across assets {previous_asset}/{asset_name}")
        suffix = source.suffix.casefold()
        if not suffix or not re.fullmatch(r"\.[a-z0-9]+", suffix):
            raise ExtensionDatasetError(f"raw art has unsafe file suffix: {source}")
        output_name = f"{asset_name}{suffix}"
        shutil.copyfile(source, art_output / output_name)
        output_record["path"] = output_name
        art_records.append(output_record)

    icon_records: list[dict[str, Any]] = []
    icon_hash_owners: dict[str, int] = {}
    for mapping in mappings:
        source = _resolve_inside(
            fixed_icon_root,
            f"{mapping.business_id}.png",
            label=f"fixed icon {mapping.business_id}",
        )
        if not source.is_file():
            raise ExtensionDatasetError(f"fixed PNG is missing for business ID {mapping.business_id}")
        width, height, mode = _verify_fixed_icon(source)
        source_sha256 = sha256_file(source)
        previous_owner = icon_hash_owners.setdefault(source_sha256, mapping.business_id)
        if previous_owner != mapping.business_id:
            raise ExtensionDatasetError(f"fixed PNG is byte-identical across business IDs {previous_owner}/{mapping.business_id}")
        for variant in mapping.asset_variants:
            class_name = _class_name(mapping, variant)
            output_name = f"{class_name}.png"
            shutil.copyfile(source, icon_output / output_name)
            icon_records.append(
                {
                    "class_name": class_name,
                    "business_id": mapping.business_id,
                    "asset_variant": variant,
                    "path": output_name,
                    "sha256": source_sha256,
                    "width": width,
                    "height": height,
                    "mode": mode,
                }
            )
    return art_records, sorted(icon_records, key=lambda row: str(row["class_name"]))


def build_extension_dataset(
    *,
    catalog_path: Path,
    pcard_path: Path,
    official_manifest_path: Path,
    raw_art_inventory_path: Path,
    raw_art_root: Path,
    fixed_icon_root: Path,
    base_gallery_manifest_path: Path,
    base_dataset_manifest_paths: Sequence[Path],
    output_dir: Path,
    dataset_revision: str,
    catalog_revision: str,
    fixed_icon_provenance_path: Path | None = None,
    published_ui_original_root: Path | None = None,
) -> ExtensionDatasetResult:
    """Build a deterministic extension dataset using frozen local inputs only."""

    if not dataset_revision.strip() or any(character in "\r\n" for character in dataset_revision):
        raise ExtensionDatasetError("dataset revision must be a non-empty single line")
    if re.fullmatch(r"[0-9a-fA-F]{40}", catalog_revision) is None:
        raise ExtensionDatasetError("catalog revision must be one immutable 40-hex Git commit")
    if (fixed_icon_provenance_path is None) != (published_ui_original_root is None):
        raise ExtensionDatasetError("fixed-icon provenance and published-UI original root must be supplied together")
    catalog_revision = catalog_revision.lower()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ExtensionDatasetError(f"output directory already exists: {output_dir}")
    if not base_dataset_manifest_paths:
        raise ExtensionDatasetError("at least one accepted base dataset manifest is required")

    gallery_manifest = _load_object(base_gallery_manifest_path, label="base gallery manifest")
    if gallery_manifest.get("schema_version") != 1:
        raise ExtensionDatasetError("unsupported base gallery manifest schema")
    source_metadata = gallery_manifest.get("source")
    if not isinstance(source_metadata, dict):
        raise ExtensionDatasetError("base gallery source metadata is missing")
    base_ids = _exact_business_ids(source_metadata)
    base_rows = _load_base_crosswalk_lineage(
        gallery_manifest,
        tuple(path.resolve() for path in base_dataset_manifest_paths),
    )
    if tuple(sorted(base_rows)) != base_ids:
        raise ExtensionDatasetError("accepted gallery ID set disagrees with its dataset lineage")

    catalog = load_catalog(catalog_path)
    catalog_ids = tuple(card.business_id for card in catalog)
    base_id_set = set(base_ids)
    if not base_id_set.issubset(catalog_ids):
        missing = sorted(base_id_set - set(catalog_ids))
        raise ExtensionDatasetError(f"accepted business IDs disappeared: {missing[:20]}")
    extension_ids = tuple(value for value in catalog_ids if value not in base_id_set)
    expected_extension_ids = tuple(range(max(base_ids) + 1, max(catalog_ids) + 1))
    if not extension_ids or extension_ids != expected_extension_ids:
        raise ExtensionDatasetError(
            "catalog delta is not a non-empty contiguous append-only extension; "
            f"actual={list(extension_ids)}, expected={list(expected_extension_ids)}"
        )

    official_manifest = _load_object(official_manifest_path, label="official game manifest")
    official_assets = _official_assets(official_manifest)
    mappings = build_crosswalk(catalog, _load_pcard(pcard_path), official_assets)
    current_by_id = {mapping.business_id: mapping for mapping in mappings}
    _compare_base_identity(base_rows, current_by_id)
    extension_mappings = tuple(current_by_id[value] for value in extension_ids)

    fixed_icon_source = {
        "repository": CATALOG_REPOSITORY,
        "commit": catalog_revision,
        "license": CATALOG_LICENSE,
    }
    if fixed_icon_provenance_path is not None:
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from tools.published_ui_reference import (
            PublishedUiReferenceError,
            validate_published_ui_source_manifest,
        )

        try:
            fixed_icon_source = validate_published_ui_source_manifest(
                fixed_icon_provenance_path,
                original_root=published_ui_original_root,
                output_root=fixed_icon_root,
                catalog_rows={
                    card.business_id: {"name": card.name, "upgraded": card.upgraded}
                    for card in catalog
                },
                component="skill_card",
                expected_ids=extension_ids,
            )
        except PublishedUiReferenceError as error:
            raise ExtensionDatasetError(f"fixed-icon provenance: {error}") from error

    raw_inventory = _load_object(raw_art_inventory_path, label="raw-art inventory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        art_records, icon_records = _copy_inputs(
            temporary=temporary,
            mappings=extension_mappings,
            official_assets=official_assets,
            raw_inventory=raw_inventory,
            raw_art_root=raw_art_root.resolve(),
            fixed_icon_root=fixed_icon_root.resolve(),
        )
        crosswalk_path = temporary / "crosswalk.json"
        _write_json(
            crosswalk_path,
            [mapping.crosswalk_row() for mapping in extension_mappings],
        )
        input_hashes = {
            "catalog_sha256": sha256_file(catalog_path),
            "decoded_pcard_sha256": sha256_file(pcard_path),
            "official_manifest_sha256": sha256_file(official_manifest_path),
            "raw_art_inventory_sha256": sha256_file(raw_art_inventory_path),
            "base_gallery_manifest_sha256": sha256_file(base_gallery_manifest_path),
            "base_dataset_manifest_sha256s": [sha256_file(path) for path in base_dataset_manifest_paths],
        }
        if fixed_icon_provenance_path is not None:
            input_hashes["fixed_icon_provenance_sha256"] = sha256_file(fixed_icon_provenance_path)
        manifest = {
            "schema_version": 1,
            "dataset_id": f"task095-skill-card-extension-{dataset_revision}",
            "dataset_revision": dataset_revision,
            "source_domain": SOURCE_DOMAIN,
            "build": {
                "mode": "OFFLINE_APPEND_ONLY_EXTENSION",
                "tool_contract": "task095-skill-card-extension-dataset-v1",
                "python_version": sys.version.split()[0],
                "pillow_version": PIL.__version__,
                "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
            },
            "scope": _business_scope(extension_ids),
            "sources": {
                **input_hashes,
                "catalog": {
                    "repository": CATALOG_REPOSITORY,
                    "commit": catalog_revision,
                    "license": CATALOG_LICENSE,
                },
                "fixed_icons": fixed_icon_source,
                "catalog_record_count": len(catalog),
                "decoded_pcard_record_count": len(mappings),
                "official_manifest_revision": official_manifest.get("revision"),
                "official_asset_server": "game-gakuen-idolmaster official asset service",
                "network_access": "NONE",
                "credential_materialized": False,
            },
            "mapping": {
                "crosswalk_path": crosswalk_path.name,
                "crosswalk_sha256": sha256_file(crosswalk_path),
                "catalog_count": len(extension_ids),
                "decoded_master_count": len(extension_ids),
                "one_to_one_count": len(extension_ids),
                "base_business_id_count": len(base_ids),
                "name_aliases": NAME_ALIASES,
            },
            "art": {
                "root": "art",
                "count": len(art_records),
                "images": art_records,
                "identical_raw_art_groups": [],
            },
            "icons": {
                "root": "icons",
                "count": len(icon_records),
                "images": icon_records,
            },
            "validation": {
                "status": "CLEAR",
                "expected_business_id_count": len(extension_ids),
                "missing_ids": [],
                "surplus_classes": [],
                "decode_failures": [],
                "unexpected_cross_id_duplicate_groups": [],
                "known_visual_ambiguities": [],
                "append_only_base_business_ids_sha256": hashlib.sha256(json.dumps(list(base_ids), separators=(",", ":")).encode("ascii"))
                .hexdigest()
                .upper(),
                "old_identity_fields_compared": list(IDENTITY_FIELDS),
            },
            "gk_img_training_source": "REJECTED",
        }
        _write_json(temporary / "dataset_manifest.json", manifest)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ExtensionDatasetResult(
        output_dir=output_dir,
        manifest=manifest,
        extension_business_ids=extension_ids,
    )


def _paths(values: Iterable[str]) -> tuple[Path, ...]:
    return tuple(Path(value) for value in values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a frozen append-only skill-card EXTENSION dataset")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--pcard", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--raw-art-inventory", type=Path, required=True)
    parser.add_argument("--raw-art-root", type=Path, required=True)
    parser.add_argument("--fixed-icon-root", type=Path, required=True)
    parser.add_argument("--fixed-icon-provenance", type=Path)
    parser.add_argument("--published-ui-original-root", type=Path)
    parser.add_argument("--base-gallery-manifest", type=Path, required=True)
    parser.add_argument(
        "--base-dataset-manifest",
        action="append",
        required=True,
        help="accepted dataset lineage in oldest-to-newest order; repeat for every layer",
    )
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--catalog-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build_extension_dataset(
            catalog_path=args.catalog,
            pcard_path=args.pcard,
            official_manifest_path=args.official_manifest,
            raw_art_inventory_path=args.raw_art_inventory,
            raw_art_root=args.raw_art_root,
            fixed_icon_root=args.fixed_icon_root,
            base_gallery_manifest_path=args.base_gallery_manifest,
            base_dataset_manifest_paths=_paths(args.base_dataset_manifest),
            output_dir=args.output_dir,
            dataset_revision=args.dataset_revision,
            catalog_revision=args.catalog_revision,
            fixed_icon_provenance_path=args.fixed_icon_provenance,
            published_ui_original_root=args.published_ui_original_root,
        )
    except ExtensionDatasetError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "manifest": str(result.output_dir / "dataset_manifest.json"),
                "extension_business_ids": list(result.extension_business_ids),
                "validation": result.manifest["validation"]["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
