from __future__ import annotations

import io
import os
import csv
import sys
import json
import math
import time
import base64
import shutil
import hashlib
import argparse
import unicodedata
from typing import Any
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image
from playwright.sync_api import Page, Locator, sync_playwright

CATALOG_REPOSITORY = "https://github.com/surisuririsu/gakumas-tools"
CATALOG_COMMIT = "3e9e8ebdc929dedd32cdd8d8911e5d8a905f76b6"
CATALOG_URL = (
    "https://raw.githubusercontent.com/surisuririsu/gakumas-tools/"
    f"{CATALOG_COMMIT}/packages/gakumas-data/csv/skill_cards.csv"
)
WEBDATA_URL = "https://gkms-webdata.idolism.org/api/pcard"
RENDERER_URL = "https://gkms.idolism.org/pcard"
RENDERER_REPOSITORY = "https://github.com/vertesan/hatsuboshi-library"
RENDERER_COMMIT = "686ae0e07e2564ff0a7beec17db13acbeb0d779d"
EXPECTED_IDS = set(range(1, 857))
TARGET_GROUPS = ((757, 758), (789, 791, 793), (790, 792, 794))
KNOWN_VISUAL_AMBIGUITIES = {
    (789, 791, 793): {"sense": 789, "logic": 791, "anomaly": 793},
    (790, 792, 794): {"sense": 790, "logic": 792, "anomaly": 794},
}
NAME_ALIASES = {
    "ストレッチ談義": "ストレッチ談議",
    "愛をこめて": "愛を込めて",
    "月明りに包まれて": "月明かりに包まれて",
    "夢と現境界線": "夢と現の境界線",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_name(name: str, upgraded: bool = False) -> str:
    value = unicodedata.normalize("NFKC", name).replace(" ", "")
    for source, destination in NAME_ALIASES.items():
        value = value.replace(source, destination)
    if upgraded and not value.endswith("+"):
        value += "+"
    return value


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

    @property
    def internal_key(self) -> tuple[str, int]:
        return self.internal_id, self.upgrade_count


def load_catalog(raw: bytes) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    ids = [int(row["id"]) for row in rows]
    if len(rows) != 856 or set(ids) != EXPECTED_IDS or len(ids) != len(set(ids)):
        raise ValueError("the pinned catalog must contain every ID from 1 through 856 exactly once")
    return rows


def manifest_asset_names(manifest: dict[str, Any]) -> set[str]:
    return {str(item["name"]) for item in manifest.get("assetBundleList", [])}


def build_crosswalk(
    catalog: list[dict[str, str]],
    api_cards: list[dict[str, Any]],
    official_asset_names: set[str],
) -> list[CardMapping]:
    api_index: dict[str, dict[str, Any]] = {}
    for card in api_cards:
        key = normalized_name(str(card["name"]))
        if key in api_index:
            raise ValueError(f"webdata contains a duplicate normalized card name: {key}")
        api_index[key] = card

    mappings: list[CardMapping] = []
    used_internal_keys: set[tuple[str, int]] = set()
    for row in catalog:
        upgraded = row["upgraded"].upper() == "TRUE"
        key = normalized_name(row["name"], upgraded)
        card = api_index.get(key)
        if card is None:
            raise ValueError(f"catalog ID {row['id']} has no unique webdata match: {row['name']}")
        internal_key = (str(card["id"]), int(card["upgradeCount"]))
        if internal_key in used_internal_keys:
            raise ValueError(f"multiple catalog IDs map to {internal_key}")
        used_internal_keys.add(internal_key)

        asset_id = str(card["assetId"])
        variants = tuple(
            sorted(
                name
                for name in official_asset_names
                if name == asset_id or name.startswith(f"{asset_id}-")
            )
        )
        is_character_asset = bool(card["isCharacterAsset"])
        if not variants:
            raise ValueError(f"official manifest has no asset for {asset_id}")
        if is_character_asset and any(name == asset_id for name in variants):
            raise ValueError(f"character-specific asset unexpectedly has an unsuffixed object: {asset_id}")
        if not is_character_asset and variants != (asset_id,):
            raise ValueError(f"non-character asset has an ambiguous official object mapping: {asset_id}")
        mappings.append(
            CardMapping(
                business_id=int(row["id"]),
                catalog_name=row["name"],
                api_name=str(card["name"]),
                internal_id=str(card["id"]),
                upgrade_count=int(card["upgradeCount"]),
                asset_id=asset_id,
                is_character_asset=is_character_asset,
                asset_variants=variants,
            )
        )

    if len(mappings) != 856 or len(used_internal_keys) != 856 or len(api_cards) != 856:
        raise ValueError("catalog/webdata join is not a complete one-to-one mapping")
    return sorted(mappings, key=lambda item: item.business_id)


def class_name(mapping: CardMapping, asset_variant: str) -> str:
    if asset_variant == mapping.asset_id:
        return str(mapping.business_id)
    suffix = asset_variant.removeprefix(f"{mapping.asset_id}-")
    if not suffix or suffix == asset_variant:
        raise ValueError(f"invalid asset variant {asset_variant} for {mapping.asset_id}")
    return f"{mapping.business_id}_{suffix}"


def download_art(
    asset_name: str,
    manifest_record: dict[str, Any],
    url_format: str,
    destination: Path,
    bundle_destination: Path,
    deobfuscator_type: type,
) -> dict[str, Any]:
    import UnityPy

    url = url_format.replace("{o}", str(manifest_record["objectName"]))
    last_error: Exception | None = None
    bundle = b""
    for attempt in range(1, 7):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            bundle = response.content
            break
        except requests.RequestException as error:
            last_error = error
            if attempt == 6:
                raise
            time.sleep(min(2 ** (attempt - 1), 16))
    if not bundle:
        raise RuntimeError(f"official bundle download failed for {asset_name}") from last_error
    if len(bundle) != int(manifest_record["size"]):
        raise ValueError(f"official bundle size mismatch for {asset_name}")
    official_md5 = hashlib.md5(bundle).hexdigest()
    if official_md5.lower() != str(manifest_record["md5"]).lower():
        raise ValueError(f"official bundle MD5 mismatch for {asset_name}")
    bundle_destination.write_bytes(bundle)
    decoded_bundle = bundle
    if not decoded_bundle.startswith(b"UnityFS"):
        decoded_bundle = deobfuscator_type(f"{asset_name}.unity3d").process(decoded_bundle)
    if not decoded_bundle.startswith(b"UnityFS"):
        raise ValueError(f"official bundle deobfuscation failed for {asset_name}")
    environment = UnityPy.load(decoded_bundle)
    textures = [obj.parse_as_object() for obj in environment.objects if obj.type.name == "Texture2D"]
    if not textures:
        raise ValueError(f"official bundle has no Texture2D for {asset_name}")
    image = max(textures, key=lambda texture: int(texture.m_CompleteImageSize)).image.convert("RGBA")
    width, height = image.size
    mode = image.mode
    image.save(destination, format="WEBP", lossless=True, method=6)
    return {
        "asset_name": asset_name,
        "url": url,
        "path": destination.name,
        "sha256": sha256_file(destination),
        "width": width,
        "height": height,
        "mode": mode,
        "official_object_name": manifest_record["objectName"],
        "official_bundle_size": len(bundle),
        "official_bundle_md5": official_md5.upper(),
        "official_bundle_sha256": sha256_bytes(bundle),
    }


def download_art_set(
    asset_names: set[str],
    asset_owners: dict[str, str],
    official_manifest: dict[str, Any],
    object_manager_root: Path,
    art_root: Path,
    bundles_root: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    object_manager_root = object_manager_root.resolve()
    if str(object_manager_root) not in sys.path:
        sys.path.insert(0, str(object_manager_root))
    import UnityPy.config
    from GkmasObjectManager.object.deobfuscate import GkmasAssetBundleDeobfuscator

    UnityPy.config.FALLBACK_UNITY_VERSION = "2022.3.21f1"
    art_root.mkdir(parents=True, exist_ok=True)
    bundles_root.mkdir(parents=True, exist_ok=True)
    manifest_index = {
        str(record["name"]): record for record in official_manifest.get("assetBundleList", [])
    }
    url_format = str(official_manifest["urlFormat"])

    def one(name: str) -> dict[str, Any]:
        destination = art_root / f"{name}.webp"
        bundle_destination = bundles_root / f"{name}.unity3d"
        record = manifest_index[name]
        if destination.is_file() and bundle_destination.is_file():
            bundle = bundle_destination.read_bytes()
            if len(bundle) != int(record["size"]) or hashlib.md5(bundle).hexdigest().lower() != str(
                record["md5"]
            ).lower():
                destination.unlink()
                bundle_destination.unlink()
                return download_art(
                    name,
                    record,
                    url_format,
                    destination,
                    bundle_destination,
                    GkmasAssetBundleDeobfuscator,
                )
            with Image.open(destination) as image:
                image.load()
                return {
                    "asset_name": name,
                    "url": url_format.replace("{o}", str(record["objectName"])),
                    "path": destination.name,
                    "sha256": sha256_file(destination),
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "official_object_name": record["objectName"],
                    "official_bundle_size": bundle_destination.stat().st_size,
                    "official_bundle_md5": hashlib.md5(bundle).hexdigest().upper(),
                    "official_bundle_sha256": sha256_file(bundle_destination),
                }
        return download_art(
            name,
            record,
            url_format,
            destination,
            bundle_destination,
            GkmasAssetBundleDeobfuscator,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(one, sorted(asset_names)))
    by_hash: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record["asset_name"])
        if record["width"] <= 0 or record["height"] <= 0:
            raise ValueError(f"undecodable or empty upstream art: {record['asset_name']}")
    identical_art_groups = []
    for image_hash, names in by_hash.items():
        if len(names) < 2:
            continue
        owners = sorted({asset_owners[name] for name in names})
        identical_art_groups.append(
            {
                "sha256": image_hash,
                "asset_ids": owners,
                "asset_variants": sorted(names),
                "cross_base_asset": len(owners) > 1,
            }
        )
    return records, identical_art_groups


def wait_for_image(page: Page, locator: Locator) -> None:
    handle = locator.element_handle()
    if handle is None:
        raise RuntimeError("card art element disappeared")
    page.wait_for_function(
        "image => image.complete && image.naturalWidth > 0 && image.naturalHeight > 0",
        arg=handle,
        timeout=30_000,
    )


def render_icons(
    mappings: list[CardMapping],
    art_root: Path,
    icons_root: Path,
    edge_path: Path,
) -> list[dict[str, Any]]:
    icons_root.mkdir(parents=True, exist_ok=True)
    by_render_key = {
        (mapping.internal_id, normalized_name(mapping.api_name)): mapping for mapping in mappings
    }
    rendered_business_ids: set[int] = set()
    image_records: list[dict[str, Any]] = []
    local_storage_filter = {
        "rarities": [],
        "planTypes": [],
        "categories": [],
        "origin": [],
        "effectTypes": [],
        "grades": [],
        "characters": [],
        "requirePLevel": False,
        "displayCustomization": False,
    }
    init_script = (
        "localStorage.setItem('produceCardFilter2', "
        f"JSON.stringify({json.dumps(local_storage_filter)}));"
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(edge_path), headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.add_init_script(init_script)
        page.goto(RENDERER_URL, wait_until="networkidle", timeout=120_000)
        page_count = math.ceil(len(mappings) / 24)
        for page_number in range(1, page_count + 1):
            cards = page.locator(".mantine-Card-root")
            expected_on_page = min(24, len(mappings) - (page_number - 1) * 24)
            if cards.count() != expected_on_page:
                raise ValueError(
                    f"renderer page {page_number} has {cards.count()} cards, expected {expected_on_page}"
                )
            for index in range(cards.count()):
                container = cards.nth(index)
                art = container.locator("img[alt^=p_card]")
                internal_id = art.get_attribute("alt")
                name = container.locator("p").first.inner_text().strip()
                mapping = by_render_key.get((str(internal_id), normalized_name(name)))
                if mapping is None:
                    raise ValueError(f"renderer card has no catalog mapping: {internal_id} / {name}")
                if mapping.business_id in rendered_business_ids:
                    raise ValueError(f"renderer emitted business ID twice: {mapping.business_id}")
                rendered_business_ids.add(mapping.business_id)
                icon = container.locator("div.aspect-square").first
                for asset_variant in mapping.asset_variants:
                    destination = icons_root / f"{class_name(mapping, asset_variant)}.webp"
                    if destination.is_file():
                        with Image.open(destination) as existing:
                            existing.verify()
                        image_records.append(
                            {
                                "class_name": destination.stem,
                                "business_id": mapping.business_id,
                                "asset_variant": asset_variant,
                                "path": destination.name,
                                "sha256": sha256_file(destination),
                                "width": 64,
                                "height": 64,
                            }
                        )
                        continue
                    encoded_art = base64.b64encode(
                        (art_root / f"{asset_variant}.webp").read_bytes()
                    ).decode("ascii")
                    art_url = f"data:image/webp;base64,{encoded_art}"
                    art.evaluate("(image, src) => { image.src = src; }", art_url)
                    wait_for_image(page, art)
                    png = icon.screenshot(type="png", animations="disabled")
                    with Image.open(io.BytesIO(png)) as source:
                        rendered = source.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
                        output = io.BytesIO()
                        rendered.save(output, format="WEBP", lossless=True, method=6)
                    destination.write_bytes(output.getvalue())
                    image_records.append(
                        {
                            "class_name": destination.stem,
                            "business_id": mapping.business_id,
                            "asset_variant": asset_variant,
                            "path": destination.name,
                            "sha256": sha256_file(destination),
                            "width": 64,
                            "height": 64,
                        }
                    )
            if page_number < page_count:
                first_alt = cards.first.locator("img[alt^=p_card]").get_attribute("alt")
                pagination = page.locator(".mantine-Pagination-root").first
                pagination.locator("button").last.click()
                page.wait_for_function(
                    "previous => document.querySelector('img[alt^=p_card]')?.getAttribute('alt') !== previous",
                    arg=first_alt,
                    timeout=30_000,
                )
        browser.close()

    if rendered_business_ids != EXPECTED_IDS:
        raise ValueError(f"renderer omitted IDs: {sorted(EXPECTED_IDS - rendered_business_ids)}")
    return sorted(image_records, key=lambda item: item["class_name"])


def validate_rendered_dataset(
    mappings: list[CardMapping], image_records: list[dict[str, Any]], icons_root: Path
) -> dict[str, Any]:
    expected_names = {
        class_name(mapping, variant)
        for mapping in mappings
        for variant in mapping.asset_variants
    }
    actual_paths = list(icons_root.glob("*.webp"))
    actual_names = {path.stem for path in actual_paths}
    if actual_names != expected_names or len(actual_paths) != len(expected_names):
        raise ValueError(
            f"rendered class set mismatch; missing={sorted(expected_names - actual_names)[:20]}, "
            f"surplus={sorted(actual_names - expected_names)[:20]}"
        )
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    record_by_name = {record["class_name"]: record for record in image_records}
    for path in actual_paths:
        with Image.open(path) as image:
            image.verify()
        record = record_by_name[path.stem]
        current_hash = sha256_file(path)
        if current_hash != record["sha256"]:
            raise ValueError(f"rendered image changed after creation: {path.name}")
        by_hash[current_hash].append(record)
    cross_id_duplicates = []
    for image_hash, records in by_hash.items():
        ids = sorted({int(record["business_id"]) for record in records})
        if len(ids) > 1:
            cross_id_duplicates.append(
                {
                    "sha256": image_hash,
                    "business_ids": ids,
                    "class_names": sorted(record["class_name"] for record in records),
                }
            )
    by_internal_id: dict[str, dict[int, CardMapping]] = defaultdict(dict)
    for mapping in mappings:
        by_internal_id[mapping.internal_id][mapping.upgrade_count] = mapping
    pair_count = 0
    pair_failures: list[dict[str, Any]] = []
    hashes_by_id: dict[int, set[str]] = defaultdict(set)
    for record in image_records:
        hashes_by_id[int(record["business_id"])].add(record["sha256"])
    for internal_id, grades in by_internal_id.items():
        if 0 not in grades or 1 not in grades:
            continue
        pair_count += 1
        normal = grades[0]
        upgraded = grades[1]
        shared_assets = set(normal.asset_variants) & set(upgraded.asset_variants)
        if not shared_assets or hashes_by_id[normal.business_id] & hashes_by_id[upgraded.business_id]:
            pair_failures.append(
                {
                    "internal_id": internal_id,
                    "normal_id": normal.business_id,
                    "upgraded_id": upgraded.business_id,
                }
            )
    target_results = []
    for group in TARGET_GROUPS:
        intersections = []
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                overlap = hashes_by_id[left] & hashes_by_id[right]
                if overlap:
                    intersections.append({"left": left, "right": right, "sha256": sorted(overlap)})
        target_results.append(
            {
                "business_ids": list(group),
                "byte_distinct": not intersections,
                "collisions": intersections,
            }
        )

    allowed_pairs = {
        frozenset((left, right))
        for group in KNOWN_VISUAL_AMBIGUITIES
        for left_index, left in enumerate(group)
        for right in group[left_index + 1 :]
    }
    unexpected_cross_id_duplicates = []
    for duplicate in cross_id_duplicates:
        ids = duplicate["business_ids"]
        pairs = {
            frozenset((left, right))
            for left_index, left in enumerate(ids)
            for right in ids[left_index + 1 :]
        }
        if not pairs or not pairs.issubset(allowed_pairs):
            unexpected_cross_id_duplicates.append(duplicate)
    status = "BLOCKED" if unexpected_cross_id_duplicates or pair_failures else "CLEAR"

    return {
        "status": status,
        "expected_business_id_count": 856,
        "rendered_image_count": len(image_records),
        "missing_ids": [],
        "surplus_classes": [],
        "decode_failures": [],
        "cross_id_duplicate_groups": cross_id_duplicates,
        "unexpected_cross_id_duplicate_groups": unexpected_cross_id_duplicates,
        "known_visual_ambiguities": [
            {
                "business_ids": list(group),
                "resolution": resolution,
                "basis": "user-confirmed official April-Fools visual reuse",
            }
            for group, resolution in KNOWN_VISUAL_AMBIGUITIES.items()
        ],
        "normal_upgraded_pair_count": pair_count,
        "normal_upgraded_distinct_pair_count": pair_count - len(pair_failures),
        "normal_upgraded_failures": pair_failures,
        "required_duplicate_id_groups": target_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the TASK-075 authoritative skill-card dataset")
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--official-manifest-label", required=True)
    parser.add_argument("--object-manager-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--webdata-basic",
        default=os.environ.get("GKMAS_WEBDATA_BASIC", ""),
        help="public Basic token used by the deployed Hatsuboshi Library frontend",
    )
    parser.add_argument(
        "--edge-path",
        type=Path,
        default=Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    )
    parser.add_argument("--download-workers", type=int, default=16)
    args = parser.parse_args()
    if not args.webdata_basic:
        parser.error("--webdata-basic or GKMAS_WEBDATA_BASIC is required")
    if not args.official_manifest.is_file():
        parser.error("--official-manifest must point to an exported official manifest JSON")
    if not (args.object_manager_root / "GkmasObjectManager").is_dir():
        parser.error("--object-manager-root must contain the pinned GkmasObjectManager package")
    if not args.edge_path.is_file():
        parser.error("Microsoft Edge executable was not found")
    if args.download_workers < 1:
        parser.error("--download-workers must be positive")

    output_dir = args.output_dir.resolve()
    sources_root = output_dir / "sources"
    art_root = output_dir / "art"
    bundles_root = output_dir / "bundles"
    icons_root = output_dir / "icons"
    sources_root.mkdir(parents=True, exist_ok=True)

    catalog_response = requests.get(CATALOG_URL, timeout=60)
    catalog_response.raise_for_status()
    catalog_raw = catalog_response.content
    catalog = load_catalog(catalog_raw)
    (sources_root / "skill_cards.csv").write_bytes(catalog_raw)

    api_response = requests.get(
        WEBDATA_URL,
        headers={"Authorization": f"Basic {args.webdata_basic}"},
        timeout=120,
    )
    api_response.raise_for_status()
    api_raw = api_response.content
    api_cards = api_response.json()
    (sources_root / "pcard.json").write_bytes(api_raw)

    renderer_response = requests.get(RENDERER_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    renderer_response.raise_for_status()
    renderer_raw = renderer_response.content
    (sources_root / "renderer-index.html").write_bytes(renderer_raw)

    official_manifest_raw = args.official_manifest.read_bytes()
    official_manifest = json.loads(official_manifest_raw.decode("utf-8-sig"))
    shutil.copyfile(args.official_manifest, sources_root / "official-dmm-manifest.json")
    mappings = build_crosswalk(catalog, api_cards, manifest_asset_names(official_manifest))
    asset_names = {variant for mapping in mappings for variant in mapping.asset_variants}
    asset_owners = {
        variant: mapping.asset_id
        for mapping in mappings
        for variant in mapping.asset_variants
    }
    art_records, identical_art_groups = download_art_set(
        asset_names,
        asset_owners,
        official_manifest,
        args.object_manager_root,
        art_root,
        bundles_root,
        args.download_workers,
    )
    image_records = render_icons(mappings, art_root, icons_root, args.edge_path)
    validation = validate_rendered_dataset(mappings, image_records, icons_root)

    crosswalk_payload = [
        {
            "business_id": mapping.business_id,
            "catalog_name": mapping.catalog_name,
            "api_name": mapping.api_name,
            "internal_id": mapping.internal_id,
            "upgrade_count": mapping.upgrade_count,
            "asset_id": mapping.asset_id,
            "is_character_asset": mapping.is_character_asset,
            "asset_variants": list(mapping.asset_variants),
        }
        for mapping in mappings
    ]
    crosswalk_path = output_dir / "crosswalk.json"
    write_json(crosswalk_path, crosswalk_payload)
    manifest = {
        "schema_version": 1,
        "dataset_id": f"task075-authoritative-{args.official_manifest_label}",
        "dataset_revision": args.official_manifest_label,
        "scope": {"business_id_min": 1, "business_id_max": 856},
        "sources": {
            "catalog": {
                "repository": CATALOG_REPOSITORY,
                "commit": CATALOG_COMMIT,
                "url": CATALOG_URL,
                "sha256": sha256_bytes(catalog_raw),
                "license": "BSD-3-Clause",
                "purpose": "stable project business IDs and Japanese card names only; no gk-img images used",
            },
            "decoded_master": {
                "url": WEBDATA_URL,
                "sha256": sha256_bytes(api_raw),
                "record_count": len(api_cards),
                "purpose": "one-to-one Japanese name/internal ID/asset ID mapping",
            },
            "official_game_manifest": {
                "label": args.official_manifest_label,
                "sha256": sha256_bytes(official_manifest_raw),
                "source_server": "https://api.asset.game-gakuen-idolmaster.jp/",
                "purpose": "authoritative existence and character-variant enumeration",
            },
            "renderer": {
                "url": RENDERER_URL,
                "index_sha256": sha256_bytes(renderer_raw),
                "repository": RENDERER_REPOSITORY,
                "commit": RENDERER_COMMIT,
                "license": "AGPL-3.0",
                "purpose": "deterministic game-style icon composition from art, frame, effects, cost and + overlay",
            },
            "official_asset_server": {
                "url_format": official_manifest["urlFormat"],
                "record_count": len(art_records),
                "redistribution": "prohibited; local validation/training only",
            },
        },
        "mapping": {
            "crosswalk_path": crosswalk_path.name,
            "crosswalk_sha256": sha256_file(crosswalk_path),
            "catalog_count": len(catalog),
            "decoded_master_count": len(api_cards),
            "one_to_one_count": len(mappings),
            "name_aliases": NAME_ALIASES,
        },
        "art": {
            "root": "art",
            "bundle_root": "bundles",
            "count": len(art_records),
            "images": art_records,
            "identical_raw_art_groups": identical_art_groups,
            "note": (
                "Raw illustration reuse is recorded but is not a dataset failure. "
                "The strict duplicate gate applies to complete rendered skill-card icons."
            ),
        },
        "icons": {"root": "icons", "count": len(image_records), "images": image_records},
        "validation": validation,
        "gk_img_training_source": "REJECTED",
    }
    manifest_path = output_dir / "dataset_manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "business_ids": len(mappings),
                "art_assets": len(art_records),
                "rendered_icons": len(image_records),
                "validation": validation,
            },
            ensure_ascii=False,
        )
    )
    return 0 if validation["status"] == "CLEAR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
