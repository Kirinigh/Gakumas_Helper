from __future__ import annotations

import io
import re
import csv
import sys
import json
import time
import hashlib
import argparse
from typing import Any
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

CATALOG_COMMIT = "3e9e8ebdc929dedd32cdd8d8911e5d8a905f76b6"
CATALOG_REPOSITORY = "https://github.com/surisuririsu/gakumas-tools"
CATALOG_ICON_FORMAT = (
    "https://raw.githubusercontent.com/surisuririsu/gakumas-tools/"
    f"{CATALOG_COMMIT}/packages/gakumas-images/images/pItems/icons/{{business_id}}.png"
)
CATALOG_ICON_PATH_FORMAT = "packages/gakumas-images/images/pItems/icons/{business_id}.png"
CATALOG_CSV_GIT_BLOB = "cc39b2098bd18cd7f6cd6a03b9ac06d6b5c216d5"
ELIGIBLE_SOURCE_TYPES = frozenset(("pIdol", "support"))
EXPECTED_CATALOG_IDS = frozenset(range(1, 469))
EXPECTED_ELIGIBLE_COUNT = 420
EXPECTED_VISUAL_IDENTITY_COUNT = 273
EXPECTED_UPGRADE_PAIR_COUNT = 145
KNOWN_PLAN_AMBIGUITIES = (
    {
        "business_ids": (406, 407, 408),
        "resolver": "plan",
        "plan_to_business_id": {"sense": 406, "logic": 407, "anomaly": 408},
    },
)
OFFICIAL_ASSET_PATTERN = re.compile(r"^img_general_pitem_([0-3])-(\d+)$")
OUTPUT_ICON_SIZE = 130
REFERENCE_ART_SIZE = 108
REFERENCE_ART_OFFSET = (OUTPUT_ICON_SIZE - REFERENCE_ART_SIZE) // 2
OPAQUE_ALPHA_THRESHOLD = 0.9


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_bool(value: str) -> bool:
    normalised = value.strip().upper()
    if normalised not in {"TRUE", "FALSE"}:
        raise ValueError(f"invalid boolean value: {value}")
    return normalised == "TRUE"


@dataclass(frozen=True)
class PItem:
    business_id: int
    name: str
    source_type: str
    upgraded: bool
    rarity: str
    plan: str
    mode: str
    p_idol_id: int | None


def load_catalog(raw: bytes) -> tuple[list[PItem], list[PItem]]:
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    ids = [int(row["id"]) for row in rows]
    if len(rows) != 468 or frozenset(ids) != EXPECTED_CATALOG_IDS or len(ids) != len(set(ids)):
        raise ValueError("the pinned catalog must contain every P-item ID 1..468 exactly once")
    items = [
        PItem(
            business_id=int(row["id"]),
            name=row["name"],
            source_type=row["sourceType"],
            upgraded=parse_bool(row["upgraded"]),
            rarity=row["rarity"],
            plan=row["plan"],
            mode=row["mode"],
            p_idol_id=int(row["pIdolId"]) if row["pIdolId"] else None,
        )
        for row in rows
    ]
    eligible = [item for item in items if item.source_type in ELIGIBLE_SOURCE_TYPES]
    excluded = [item for item in items if item.source_type not in ELIGIBLE_SOURCE_TYPES]
    if len(eligible) != EXPECTED_ELIGIBLE_COUNT or any(item.source_type != "produce" for item in excluded):
        raise ValueError("the pinned non-produce P-item scope must contain 420 IDs and exclude 48 produce IDs")
    return sorted(eligible, key=lambda item: item.business_id), sorted(
        excluded, key=lambda item: item.business_id
    )


def load_tree_index(raw: bytes) -> dict[str, dict[str, Any]]:
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("truncated"):
        raise ValueError("the pinned GitHub tree response is truncated")
    return {str(record["path"]): record for record in payload.get("tree", [])}


def composite_rgb(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def reference_art_patch(image: Image.Image) -> np.ndarray:
    normalised = image.convert("RGB").resize(
        (OUTPUT_ICON_SIZE, OUTPUT_ICON_SIZE), Image.Resampling.LANCZOS
    )
    rgb = np.asarray(normalised, dtype=np.float32) / 255.0
    start = REFERENCE_ART_OFFSET
    end = start + REFERENCE_ART_SIZE
    return rgb[start:end, start:end]


def official_reference_render(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    rgba = image.convert("RGBA").resize(
        (REFERENCE_ART_SIZE, REFERENCE_ART_SIZE), Image.Resampling.LANCZOS
    )
    values = np.asarray(rgba, dtype=np.float32) / 255.0
    opaque = values[:, :, 3] > OPAQUE_ALPHA_THRESHOLD
    if int(opaque.sum()) < 64:
        raise ValueError("official P-item art has too little opaque foreground")
    return values[:, :, :3], opaque


def canonical_image_sha256(image: Image.Image) -> str:
    resized = composite_rgb(image).resize((OUTPUT_ICON_SIZE, OUTPUT_ICON_SIZE), Image.Resampling.LANCZOS)
    return sha256_bytes(np.asarray(resized, dtype=np.uint8).tobytes())


def download_reference_icons(
    items: list[PItem],
    tree_index: dict[str, dict[str, Any]],
    destination: Path,
    workers: int,
) -> list[dict[str, Any]]:
    import requests

    destination.mkdir(parents=True, exist_ok=True)

    def one(item: PItem) -> dict[str, Any]:
        repository_path = CATALOG_ICON_PATH_FORMAT.format(business_id=item.business_id)
        tree_record = tree_index.get(repository_path)
        if tree_record is None or tree_record.get("type") != "blob":
            raise ValueError(f"pinned repository has no P-item icon for ID {item.business_id}")
        expected_git_sha = str(tree_record["sha"])
        target = destination / f"{item.business_id}.png"
        data = target.read_bytes() if target.is_file() else b""
        if git_blob_sha1(data) != expected_git_sha:
            response = requests.get(
                CATALOG_ICON_FORMAT.format(business_id=item.business_id),
                timeout=60,
            )
            response.raise_for_status()
            data = response.content
            if git_blob_sha1(data) != expected_git_sha:
                raise ValueError(f"Git blob SHA mismatch for reference icon {item.business_id}")
            target.write_bytes(data)
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height, mode = image.width, image.height, image.mode
            canonical_sha = canonical_image_sha256(image)
        return {
            "business_id": item.business_id,
            "path": target.name,
            "repository_path": repository_path,
            "git_blob_sha1": expected_git_sha,
            "sha256": sha256_bytes(data),
            "canonical_sha256": canonical_sha,
            "width": width,
            "height": height,
            "mode": mode,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return sorted(executor.map(one, items), key=lambda record: int(record["business_id"]))


def official_asset_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in manifest.get("assetBundleList", []):
        name = str(record["name"])
        if OFFICIAL_ASSET_PATTERN.fullmatch(name):
            if name in records:
                raise ValueError(f"duplicate official P-item asset: {name}")
            records[name] = record
    counts = defaultdict(int)
    for name in records:
        counts[OFFICIAL_ASSET_PATTERN.fullmatch(name).group(1)] += 1  # type: ignore[union-attr]
    if dict(counts) != {"0": 14, "1": 65, "2": 127, "3": 301}:
        raise ValueError(f"unexpected official P-item asset counts: {dict(counts)}")
    return records


def download_official_art(
    records: dict[str, dict[str, Any]],
    url_format: str,
    object_manager_root: Path,
    art_root: Path,
    bundle_root: Path,
    workers: int,
) -> list[dict[str, Any]]:
    import requests

    object_manager_root = object_manager_root.resolve()
    if str(object_manager_root) not in sys.path:
        sys.path.insert(0, str(object_manager_root))
    import UnityPy
    import UnityPy.config
    from GkmasObjectManager.object.deobfuscate import GkmasAssetBundleDeobfuscator

    UnityPy.config.FALLBACK_UNITY_VERSION = "2022.3.21f1"
    art_root.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)

    def one(name: str) -> dict[str, Any]:
        record = records[name]
        url = url_format.replace("{o}", str(record["objectName"]))
        bundle_path = bundle_root / f"{name}.unity3d"
        art_path = art_root / f"{name}.webp"
        bundle = bundle_path.read_bytes() if bundle_path.is_file() else b""
        expected_size = int(record["size"])
        expected_md5 = str(record["md5"]).lower()
        if len(bundle) != expected_size or hashlib.md5(bundle).hexdigest().lower() != expected_md5:
            last_error: Exception | None = None
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
                raise RuntimeError(f"official bundle download failed for {name}") from last_error
            if len(bundle) != expected_size or hashlib.md5(bundle).hexdigest().lower() != expected_md5:
                raise ValueError(f"official bundle integrity mismatch for {name}")
            bundle_path.write_bytes(bundle)
            decoded = bundle
            if not decoded.startswith(b"UnityFS"):
                decoded = GkmasAssetBundleDeobfuscator(f"{name}.unity3d").process(decoded)
            if not decoded.startswith(b"UnityFS"):
                raise ValueError(f"official bundle deobfuscation failed for {name}")
            environment = UnityPy.load(decoded)
            textures = [obj.parse_as_object() for obj in environment.objects if obj.type.name == "Texture2D"]
            if not textures:
                raise ValueError(f"official bundle has no Texture2D for {name}")
            image = max(textures, key=lambda texture: int(texture.m_CompleteImageSize)).image.convert("RGBA")
            image.save(art_path, format="WEBP", lossless=True, method=6)
        with Image.open(art_path) as image:
            image.load()
            width, height, mode = image.width, image.height, image.mode
            canonical_sha = canonical_image_sha256(image)
        return {
            "asset_name": name,
            "path": art_path.name,
            "sha256": sha256_file(art_path),
            "canonical_sha256": canonical_sha,
            "width": width,
            "height": height,
            "mode": mode,
            "official_object_name": record["objectName"],
            "official_bundle_size": len(bundle),
            "official_bundle_md5": hashlib.md5(bundle).hexdigest().upper(),
            "official_bundle_sha256": sha256_bytes(bundle),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return sorted(executor.map(one, sorted(records)), key=lambda record: str(record["asset_name"]))


def masked_foreground_mse(
    reference: np.ndarray,
    official_rgb: np.ndarray,
    official_opaque: np.ndarray,
) -> np.ndarray:
    difference = official_rgb - reference[None, :, :, :]
    weighted = np.square(difference) * official_opaque[:, :, :, None]
    denominator = official_opaque.sum(axis=(1, 2), dtype=np.float64) * 3.0
    return weighted.sum(axis=(1, 2, 3), dtype=np.float64) / denominator


def map_business_ids(
    items: list[PItem],
    reference_root: Path,
    art_root: Path,
    art_records: list[dict[str, Any]],
    *,
    maximum_mse: float,
    minimum_margin: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = sorted(art_records, key=lambda record: str(record["asset_name"]))
    reference_patches = []
    for item in items:
        with Image.open(reference_root / f"{item.business_id}.png") as image:
            reference_patches.append(reference_art_patch(image))
    official_rgb = []
    official_opaque = []
    for record in candidates:
        with Image.open(art_root / str(record["path"])) as image:
            rgb, opaque = official_reference_render(image)
            official_rgb.append(rgb)
            official_opaque.append(opaque)
    official_rgb_array = np.stack(official_rgb).astype(np.float32)
    official_opaque_array = np.stack(official_opaque)
    mappings: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row_index, item in enumerate(items):
        distances = masked_foreground_mse(
            reference_patches[row_index], official_rgb_array, official_opaque_array
        )
        order = np.argsort(distances, kind="stable")
        best_index, second_index = int(order[0]), int(order[1])
        best = candidates[best_index]
        best_mse = float(distances[best_index])
        second_mse = float(distances[second_index])
        margin = second_mse - best_mse
        mapping = {
            **asdict(item),
            "asset_name": best["asset_name"],
            "asset_sha256": best["sha256"],
            "reference_match_mse": best_mse,
            "reference_match_margin": margin,
            "top5_asset_candidates": [
                {
                    "asset_name": candidates[int(index)]["asset_name"],
                    "mse": float(distances[int(index)]),
                }
                for index in order[:5]
            ],
        }
        mappings.append(mapping)
        if best_mse > maximum_mse or margin < minimum_margin:
            failures.append(mapping)
    return sorted(mappings, key=lambda record: int(record["business_id"])), failures


def materialise_icons(
    mappings: list[dict[str, Any]],
    art_root: Path,
    destination: Path,
) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for mapping in mappings:
        business_id = int(mapping["business_id"])
        target = destination / f"{business_id}.webp"
        with Image.open(art_root / f"{mapping['asset_name']}.webp") as image:
            icon = composite_rgb(image).resize(
                (OUTPUT_ICON_SIZE, OUTPUT_ICON_SIZE), Image.Resampling.LANCZOS
            )
            icon.save(target, format="WEBP", lossless=True, method=6)
        with Image.open(target) as image:
            image.verify()
        records.append(
            {
                "business_id": business_id,
                "path": target.name,
                "sha256": sha256_file(target),
                "width": OUTPUT_ICON_SIZE,
                "height": OUTPUT_ICON_SIZE,
            }
        )
    return records


def audit_dataset(
    items: list[PItem],
    mappings: list[dict[str, Any]],
    icon_records: list[dict[str, Any]],
    mapping_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_ids = {item.business_id for item in items}
    mapped_ids = {int(record["business_id"]) for record in mappings}
    icon_ids = {int(record["business_id"]) for record in icon_records}
    by_hash: dict[str, list[int]] = defaultdict(list)
    for record in icon_records:
        by_hash[str(record["sha256"])].append(int(record["business_id"]))
    duplicate_groups = [
        {"business_ids": sorted(ids), "sha256": image_hash}
        for image_hash, ids in by_hash.items()
        if len(ids) > 1
    ]
    by_p_idol: dict[int, list[PItem]] = defaultdict(list)
    for item in items:
        if item.source_type == "pIdol" and item.p_idol_id is not None:
            by_p_idol[item.p_idol_id].append(item)
    pairs = []
    pair_failures = []
    hash_by_id = {int(record["business_id"]): str(record["sha256"]) for record in icon_records}
    mapping_by_id = {int(record["business_id"]): record for record in mappings}
    expected_ambiguity_groups = [
        set(record["business_ids"]) for record in KNOWN_PLAN_AMBIGUITIES
    ]
    for p_idol_id, members in sorted(by_p_idol.items()):
        normal = [item for item in members if not item.upgraded]
        upgraded = [item for item in members if item.upgraded]
        if len(normal) == 1 and len(upgraded) == 1:
            record = {
                "p_idol_id": p_idol_id,
                "normal_id": normal[0].business_id,
                "upgraded_id": upgraded[0].business_id,
                "asset_name": mapping_by_id[normal[0].business_id]["asset_name"],
                "same_visual_asset": (
                    mapping_by_id[normal[0].business_id]["asset_name"]
                    == mapping_by_id[upgraded[0].business_id]["asset_name"]
                ),
            }
            pairs.append(record)
            expected_ambiguity_groups.append(
                {normal[0].business_id, upgraded[0].business_id}
            )
            if not record["same_visual_asset"]:
                pair_failures.append(record)
        elif len(normal) != 1 or upgraded:
            pair_failures.append(
                {
                    "p_idol_id": p_idol_id,
                    "normal_ids": [item.business_id for item in normal],
                    "upgraded_ids": [item.business_id for item in upgraded],
                }
            )
    asset_owners: dict[str, list[int]] = defaultdict(list)
    for mapping in mappings:
        asset_owners[str(mapping["asset_name"])].append(int(mapping["business_id"]))
    conflicting_asset_reuse = [
        {"asset_name": name, "business_ids": sorted(ids)}
        for name, ids in asset_owners.items()
        if len(ids) > 1 and set(ids) not in expected_ambiguity_groups
    ]
    unexpected_duplicate_groups = [
        group
        for group in duplicate_groups
        if set(group["business_ids"]) not in expected_ambiguity_groups
    ]
    clear = (
        mapped_ids == expected_ids
        and icon_ids == expected_ids
        and not mapping_failures
        and not pair_failures
        and not conflicting_asset_reuse
        and not unexpected_duplicate_groups
        and len(by_hash) == EXPECTED_VISUAL_IDENTITY_COUNT
        and len(pairs) == EXPECTED_UPGRADE_PAIR_COUNT
    )
    return {
        "status": "CLEAR" if clear else "BLOCKED",
        "expected_business_id_count": len(expected_ids),
        "mapped_business_id_count": len(mapped_ids),
        "missing_ids": sorted(expected_ids - mapped_ids),
        "surplus_ids": sorted(mapped_ids - expected_ids),
        "decode_failures": [],
        "mapping_failures": mapping_failures,
        "known_visual_ambiguity_groups": sorted(
            duplicate_groups, key=lambda group: int(group["business_ids"][0])
        ),
        "unexpected_cross_id_duplicate_groups": unexpected_duplicate_groups,
        "visual_identity_count": len(by_hash),
        "normal_upgraded_pair_count": len(pairs),
        "normal_upgraded_same_visual_pair_count": sum(
            pair["same_visual_asset"] for pair in pairs
        ),
        "normal_upgraded_failures": pair_failures,
        "single_state_p_idol_count": sum(len(members) == 1 for members in by_p_idol.values()),
        "deterministic_context_resolvers": list(KNOWN_PLAN_AMBIGUITIES),
        "conflicting_official_asset_reuse": conflicting_asset_reuse,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the TASK-085 authoritative P-item dataset")
    parser.add_argument("--catalog-csv", type=Path, required=True)
    parser.add_argument("--catalog-tree", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--official-manifest-label", required=True)
    parser.add_argument("--object-manager-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--maximum-match-mse", type=float, default=0.06)
    parser.add_argument("--minimum-match-margin", type=float, default=0.003)
    args = parser.parse_args()
    if args.download_workers < 1:
        parser.error("--download-workers must be positive")
    if args.maximum_match_mse <= 0.0 or args.minimum_match_margin < 0.0:
        parser.error("invalid match thresholds")
    if not (args.object_manager_root / "GkmasObjectManager").is_dir():
        parser.error("--object-manager-root must contain the pinned GkmasObjectManager package")

    catalog_raw = args.catalog_csv.read_bytes()
    if git_blob_sha1(catalog_raw) != CATALOG_CSV_GIT_BLOB:
        raise ValueError("catalog CSV differs from the pinned Git blob")
    tree_raw = args.catalog_tree.read_bytes()
    tree_index = load_tree_index(tree_raw)
    items, excluded = load_catalog(catalog_raw)
    manifest_raw = args.official_manifest.read_bytes()
    official_manifest = json.loads(manifest_raw.decode("utf-8-sig"))
    official_records = official_asset_records(official_manifest)

    output = args.output_dir.resolve()
    sources_root = output / "sources"
    reference_root = output / "reference-icons"
    art_root = output / "official-art"
    bundle_root = output / "official-bundles"
    icons_root = output / "icons"
    sources_root.mkdir(parents=True, exist_ok=True)
    (sources_root / "p_items.csv").write_bytes(catalog_raw)
    (sources_root / "gakumas-tools-tree.json").write_bytes(tree_raw)
    (sources_root / "official-dmm-manifest.json").write_bytes(manifest_raw)

    reference_records = download_reference_icons(
        items, tree_index, reference_root, args.download_workers
    )
    art_records = download_official_art(
        official_records,
        str(official_manifest["urlFormat"]),
        args.object_manager_root,
        art_root,
        bundle_root,
        args.download_workers,
    )
    mappings, mapping_failures = map_business_ids(
        items,
        reference_root,
        art_root,
        art_records,
        maximum_mse=args.maximum_match_mse,
        minimum_margin=args.minimum_match_margin,
    )
    crosswalk_path = output / "crosswalk.json"
    write_json(crosswalk_path, mappings)
    icon_records = materialise_icons(mappings, art_root, icons_root)
    validation = audit_dataset(items, mappings, icon_records, mapping_failures)
    manifest = {
        "schema_version": 1,
        "dataset_id": f"task085-authoritative-{args.official_manifest_label}",
        "dataset_revision": args.official_manifest_label,
        "scope": {
            "rule": "sourceType != produce",
            "eligible_source_types": sorted(ELIGIBLE_SOURCE_TYPES),
            "business_id_count": len(items),
            "excluded_produce_id_count": len(excluded),
            "business_ids": [item.business_id for item in items],
        },
        "sources": {
            "catalog": {
                "repository": CATALOG_REPOSITORY,
                "commit": CATALOG_COMMIT,
                "csv_sha256": sha256_bytes(catalog_raw),
                "csv_git_blob_sha1": CATALOG_CSV_GIT_BLOB,
                "tree_sha256": sha256_bytes(tree_raw),
                "license": "BSD-3-Clause",
                "purpose": "stable business IDs, names, source type and fixed reference icon mapping",
            },
            "official_game_manifest": {
                "label": args.official_manifest_label,
                "sha256": sha256_bytes(manifest_raw),
                "source_server": "https://api.asset.game-gakuen-idolmaster.jp/",
                "p_item_asset_count": len(official_records),
            },
            "official_asset_server": {
                "url_format": official_manifest["urlFormat"],
                "redistribution": "prohibited; local validation/training only",
            },
            "mapping_method": (
                "fixed-revision 130x130 business icon matched to the official DMM asset rendered "
                "at 108x108 in the centred art region; RGB MSE is measured only on opaque official "
                "foreground pixels across every official P-item asset; training icons are then "
                "regenerated from official art; sourceType is not treated as an asset-kind key"
            ),
        },
        "reference_icons": {
            "root": reference_root.name,
            "count": len(reference_records),
            "images": reference_records,
            "training_source": True,
            "training_role": (
                "verified rendered-domain adaptation after authoritative official-art mapping; "
                "not an independent business-ID or artwork-identity authority"
            ),
        },
        "official_art": {"root": art_root.name, "count": len(art_records), "images": art_records},
        "mapping": {
            "crosswalk_path": crosswalk_path.name,
            "crosswalk_sha256": sha256_file(crosswalk_path),
            "one_to_one_business_id_count": len(mappings),
            "maximum_match_mse": args.maximum_match_mse,
            "minimum_match_margin": args.minimum_match_margin,
        },
        "icons": {"root": icons_root.name, "count": len(icon_records), "images": icon_records},
        "validation": validation,
        "gk_img_training_source": "REJECTED",
    }
    manifest_path = output / "dataset_manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "eligible_business_ids": len(items),
                "official_assets": len(art_records),
                "visual_identities": validation["visual_identity_count"],
                "mapping_failures": len(validation["mapping_failures"]),
                "unexpected_duplicates": len(
                    validation["unexpected_cross_id_duplicate_groups"]
                ),
                "status": validation["status"],
            },
            ensure_ascii=True,
        )
    )
    return 0 if validation["status"] == "CLEAR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
