from __future__ import annotations

import sys
import json
import shutil
import hashlib
import argparse
from typing import Any
from pathlib import Path

import PIL
import numpy as np
from PIL import Image

BASE_REFERENCE_COUNT = 424
FINAL_REFERENCE_COUNT = 426
BACKGROUND_SAMPLE_COUNT = 177
TARGET_SIZE = (130, 130)
ART_SIZE = (108, 108)
ART_OFFSET = (11, 11)
MINIMUM_CLEAR_BACKGROUND_SAMPLES = 5
MARKER_SOURCE_IDS = (473, 474)
MARKER_BOX = (104, 0, 130, 26)
MARKER_DIFFERENCE_THRESHOLD = 15
MARKER_PIXEL_COUNT = 510
MARKER_PIXEL_COUNTS = {(473, 474): 510, (469, 470): 517}

BASE_SOURCE_ICONS_SHA256 = (
    "94095BA78CAEDB21748311768B0E2FAC42B451F4182057BD2D2D9916D54A6D95"
)
CROSSWALK_SHA256 = (
    "9E7CCC422C9308F20CB157DAF922E048CB49CE03B1C8EA581EA1C60BB9544182"
)
OFFICIAL_MANIFEST_CACHE_SHA256 = (
    "1C9C646698B971AB6E14FC90FE81B5E6BA22DB6EDE87ECF56599B3378C471B0B"
)
ACTIVE_CATALOG_SHA256 = (
    "FB13629F06837C852B0887E7D50B2ADAF67573E444109DEFE173C1F825DB005B"
)
ACTIVE_RIS_REVISION = "64c3d536daaf74c3ef9095a877f878ac004e71b1"
PINNED_UNITYPY_VERSION = "1.25.3"
PINNED_PILLOW_VERSION = "11.3.0"
PINNED_NUMPY_VERSION = "2.5.2"

OFFICIAL_ASSET = {
    "asset_name": "img_general_pitem_3-341",
    "object_id": 24617,
    "object_name": "cLsP8b",
    "size": 335805,
    "md5": "8826689114080CDBECA0F175DAE284F3",
    "sha256": "E5E8DBEF00AE434823D38B89BFC85941921309D457A61816A3FA453E27DBEA05",
    "rgba_pixel_sha256": (
        "90233CB2B3F6C4A84F139F732B1B5CA68FCB82021F8584BAE1B56C27280338CC"
    ),
    "width": 1024,
    "height": 1024,
}

VALIDATION_ASSETS = {
    "img_general_pitem_3-339": {
        "asset_name": "img_general_pitem_3-339",
        "object_id": 24248,
        "object_name": "RIzrq0",
        "size": 338049,
        "md5": "5C17384CC9E6AA46A83BF2B06E72DF84",
        "sha256": "D2F13EF09AB11987A53AE207A724C18D6625D71E5C5B2870B4D19B55B1459D5C",
        "rgba_pixel_sha256": (
            "58505F9A52534A32DD81853AD309314D76AEC1BF4EB19895383E0147BAE68D92"
        ),
        "width": 1024,
        "height": 1024,
    },
    "img_general_pitem_3-340": {
        "asset_name": "img_general_pitem_3-340",
        "object_id": 24346,
        "object_name": "PM2ZfG",
        "size": 200965,
        "md5": "457C37BE5B9B0F9920F0B7AEABFEB0C1",
        "sha256": "70AFB75A70AE8A9C5F84BA81FB35DB646F9F95C2B04C5EB6BB8DA794B6E9E9C0",
        "rgba_pixel_sha256": (
            "3E868E321CA0318F8EEB01CEC31478DBEA02F8D16F3AE4F3D50926EB03B227E8"
        ),
        "width": 1024,
        "height": 1024,
    },
}

EXPECTED_CATALOG_ROWS = {
    475: {
        "name": "夏の足音",
        "plan": "sense",
        "sourceType": "pIdol",
        "mode": "stage",
        "upgraded": False,
        "pIdolId": 151,
        "rarity": "SSR",
    },
    476: {
        "name": "夏の足音+",
        "plan": "sense",
        "sourceType": "pIdol",
        "mode": "stage",
        "upgraded": True,
        "pIdolId": 151,
        "rarity": "SSR",
    },
    477: {
        "name": "真夏に咲く太陽",
        "plan": "sense",
        "sourceType": "support",
        "mode": "produce",
        "upgraded": False,
        "rarity": "SSR",
    },
}

EXPECTED_OUTPUTS = {
    475: {
        "png_sha256": (
            "BE624A1B3EC6260BF45C65E62ED2E002274E8BF6ECEA31161BA8BAAE2DB079D6"
        ),
        "rgb_pixel_sha256": (
            "E69AB96D18DACE3D87AC497098CA33D06C2FCBF9503631879B8A284473B693C5"
        ),
        "size": 23088,
    },
    476: {
        "png_sha256": (
            "7047937AAABAB525871155212BA9DC154E2DF6A7ABBB66D3FDA1C26233CB3F96"
        ),
        "rgb_pixel_sha256": (
            "84F49FDA2D06F721067B35DF239C4F2CE8B36157B5EF55AAB5DE8AD9EA8C26CD"
        ),
        "size": 23631,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the audited rendered-domain references for P-item IDs 475/476"
        )
    )
    parser.add_argument("--base-source-icons", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--official-art-root", type=Path, required=True)
    parser.add_argument("--official-manifest-cache", type=Path, required=True)
    parser.add_argument("--active-catalog", type=Path, required=True)
    parser.add_argument("--official-bundle", type=Path, required=True)
    parser.add_argument("--validation-bundle-339", type=Path, required=True)
    parser.add_argument("--validation-bundle-340", type=Path, required=True)
    parser.add_argument("--object-manager-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("ascii"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest().upper()


def _numeric_pngs(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.glob("*.png") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_catalog(path: Path) -> None:
    if _sha256_file(path) != ACTIVE_CATALOG_SHA256:
        raise RuntimeError("active P-item catalog SHA-256 does not match the frozen RIS input")
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise RuntimeError("active P-item catalog must be a JSON array")
    by_id = {int(item["id"]): item for item in payload}
    for business_id, expected in EXPECTED_CATALOG_ROWS.items():
        item = by_id.get(business_id)
        if item is None or any(item.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"active catalog row {business_id} does not match the frozen input")


def _extract_official_art(
    bundle_path: Path,
    object_manager_root: Path,
    asset: dict[str, Any],
) -> Image.Image:
    raw = bundle_path.read_bytes()
    if len(raw) != asset["size"]:
        raise RuntimeError("official P-item bundle size does not match manifest revision 45")
    if hashlib.md5(raw).hexdigest().upper() != asset["md5"]:
        raise RuntimeError("official P-item bundle MD5 does not match manifest revision 45")
    if _sha256_bytes(raw) != asset["sha256"]:
        raise RuntimeError("official P-item bundle SHA-256 does not match the frozen input")

    manager_root = str(object_manager_root.resolve())
    if manager_root not in sys.path:
        sys.path.insert(0, manager_root)
    try:
        import UnityPy
        import UnityPy.config
        from GkmasObjectManager.object.deobfuscate import (
            GkmasAssetBundleDeobfuscator,
        )
    except ImportError as error:
        raise RuntimeError("UnityPy/GkmasObjectManager is required to decode the bundle") from error
    if str(getattr(UnityPy, "__version__", "")) != PINNED_UNITYPY_VERSION:
        raise RuntimeError("UnityPy version does not match the frozen render toolchain")

    UnityPy.config.FALLBACK_UNITY_VERSION = "2022.3.21f1"
    decoded = raw
    if not decoded.startswith(b"UnityFS"):
        decoded = GkmasAssetBundleDeobfuscator(
            f"{asset['asset_name']}.unity3d"
        ).process(decoded)
    if not decoded.startswith(b"UnityFS"):
        raise RuntimeError("official P-item bundle deobfuscation failed")
    environment = UnityPy.load(decoded)
    textures = [
        obj.parse_as_object()
        for obj in environment.objects
        if obj.type.name == "Texture2D"
    ]
    if not textures:
        raise RuntimeError("official P-item bundle contains no Texture2D")
    image = max(
        textures,
        key=lambda texture: int(texture.m_CompleteImageSize),
    ).image.convert("RGBA")
    if image.size != (asset["width"], asset["height"]):
        raise RuntimeError("official P-item art dimensions do not match the frozen input")
    if _sha256_bytes(image.tobytes()) != asset["rgba_pixel_sha256"]:
        raise RuntimeError("official P-item art pixels do not match the frozen input")
    return image


def _load_background_inputs(
    source_root: Path,
    crosswalk_path: Path,
    official_art_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = _numeric_pngs(source_root)
    if len(paths) != BASE_REFERENCE_COUNT:
        raise RuntimeError(
            f"expected {BASE_REFERENCE_COUNT} base references, found {len(paths)}"
        )
    if _source_digest(paths) != BASE_SOURCE_ICONS_SHA256:
        raise RuntimeError("base rendered-reference digest does not match the frozen input")
    if _sha256_file(crosswalk_path) != CROSSWALK_SHA256:
        raise RuntimeError("official-art crosswalk digest does not match the frozen input")
    payload = _read_json(crosswalk_path)
    if not isinstance(payload, list):
        raise RuntimeError("official-art crosswalk must be a JSON array")
    rows = [
        row
        for row in payload
        if row.get("rarity") == "SSR" and row.get("upgraded") is False
    ]
    if len(rows) != BACKGROUND_SAMPLE_COUNT:
        raise RuntimeError(
            f"expected {BACKGROUND_SAMPLE_COUNT} unupgraded SSR background samples"
        )

    rendered: list[np.ndarray] = []
    official: list[np.ndarray] = []
    business_ids: list[int] = []
    for row in rows:
        business_id = int(row["business_id"])
        reference_path = source_root / f"{business_id}.png"
        art_path = official_art_root / f"{row['asset_name']}.webp"
        if _sha256_file(art_path) != str(row["asset_sha256"]).upper():
            raise RuntimeError(f"official art for background sample {business_id} drifted")
        with Image.open(reference_path) as image:
            rendered.append(
                np.asarray(
                    image.convert("RGB").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                )
            )
        with Image.open(art_path) as image:
            official.append(
                np.asarray(
                    image.convert("RGBA").resize(ART_SIZE, Image.Resampling.LANCZOS)
                )
            )
        business_ids.append(business_id)
    return np.asarray(business_ids, dtype=np.int32), np.stack(rendered), np.stack(official)


def _recover_background(
    rendered: np.ndarray,
    official: np.ndarray,
    *,
    expected_minimum_clear_count: int | None = None,
) -> np.ndarray:
    background = np.rint(np.median(rendered, axis=0)).astype(np.uint8)
    minimum_clear_count = int(rendered.shape[0])
    offset_x, offset_y = ART_OFFSET
    for y in range(ART_SIZE[1]):
        for x in range(ART_SIZE[0]):
            clear = official[:, y, x, 3] <= 255 * 0.01
            clear_count = int(np.count_nonzero(clear))
            minimum_clear_count = min(minimum_clear_count, clear_count)
            if clear_count < MINIMUM_CLEAR_BACKGROUND_SAMPLES:
                raise RuntimeError(
                    f"insufficient transparent samples at official-art pixel {(x, y)}"
                )
            background[y + offset_y, x + offset_x] = np.rint(
                np.median(rendered[clear, y + offset_y, x + offset_x], axis=0)
            ).astype(np.uint8)
    if (
        expected_minimum_clear_count is not None
        and minimum_clear_count != expected_minimum_clear_count
    ):
        raise RuntimeError("background clear-sample floor drifted")
    return background


def _compose_reference(background: np.ndarray, art: Image.Image) -> np.ndarray:
    canvas = Image.fromarray(background).convert("RGBA")
    layer = Image.new("RGBA", TARGET_SIZE)
    layer.alpha_composite(
        art.resize(ART_SIZE, Image.Resampling.LANCZOS),
        dest=ART_OFFSET,
    )
    canvas.alpha_composite(layer)
    return np.asarray(canvas.convert("RGB"))


def _apply_upgrade_marker(
    image: np.ndarray,
    source_root: Path,
    source_ids: tuple[int, int] = MARKER_SOURCE_IDS,
) -> np.ndarray:
    marker_sources = []
    for business_id in source_ids:
        with Image.open(source_root / f"{business_id}.png") as source:
            marker_sources.append(
                np.asarray(
                    source.convert("RGB").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                )
            )
    normal, upgraded = marker_sources
    marker = np.max(
        np.abs(upgraded.astype(np.int16) - normal.astype(np.int16)),
        axis=2,
    ) > MARKER_DIFFERENCE_THRESHOLD
    left, top, right, bottom = MARKER_BOX
    allowed = np.zeros(marker.shape, dtype=bool)
    allowed[top:bottom, left:right] = True
    marker &= allowed
    expected_marker_count = MARKER_PIXEL_COUNTS.get(source_ids)
    if expected_marker_count is None or int(np.count_nonzero(marker)) != expected_marker_count:
        raise RuntimeError(f"fixed {source_ids!r} upgrade-marker evidence drifted")
    result = image.copy()
    result[marker] = upgraded[marker]
    return result


def _normalised_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_float = left.astype(np.float64)
    right_float = right.astype(np.float64)
    left_float -= left_float.mean(axis=(0, 1), keepdims=True)
    right_float -= right_float.mean(axis=(0, 1), keepdims=True)
    denominator = np.sqrt(
        np.square(left_float).sum(dtype=np.float64)
        * np.square(right_float).sum(dtype=np.float64)
    )
    if denominator <= 0:
        raise RuntimeError("leave-one-out correlation has no variance")
    return float((left_float * right_float).sum(dtype=np.float64) / denominator)


def _resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))


def _validate_leave_one_out(
    background: np.ndarray,
    background_business_ids: np.ndarray,
    source_root: Path,
    art_339: Image.Image,
    art_340: Image.Image,
) -> dict[str, dict[str, Any]]:
    validation_ids = {469, 470, 473, 474}
    if validation_ids.intersection(int(value) for value in background_business_ids):
        raise RuntimeError("leave-one-out IDs leaked into background training inputs")
    synthetic_469 = _compose_reference(background, art_339)
    synthetic_470 = _apply_upgrade_marker(
        synthetic_469,
        source_root,
        (473, 474),
    )
    synthetic_473 = _compose_reference(background, art_340)
    synthetic_474 = _apply_upgrade_marker(
        synthetic_473,
        source_root,
        (469, 470),
    )
    synthetic = {
        469: synthetic_469,
        470: synthetic_470,
        473: synthetic_473,
        474: synthetic_474,
    }
    real: dict[int, np.ndarray] = {}
    for path in _numeric_pngs(source_root):
        with Image.open(path) as image:
            real[int(path.stem)] = np.asarray(
                image.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
            )

    results: dict[str, dict[str, Any]] = {}
    for business_id, replacement in synthetic.items():
        query = real[business_id]
        candidates = dict(real)
        candidates[business_id] = _resize_rgb(replacement, (64, 64))
        ranking = sorted(
            (
                (_normalised_correlation(query, candidate), candidate_id)
                for candidate_id, candidate in candidates.items()
            ),
            key=lambda value: (-value[0], value[1]),
        )
        top1_similarity, top1_id = ranking[0]
        top2_similarity, top2_id = ranking[1]
        margin = top1_similarity - top2_similarity
        if top1_id != business_id or top1_similarity < 0.98 or margin < 0.02:
            raise RuntimeError(
                f"leave-one-out validation failed for rendered reference {business_id}"
            )
        results[str(business_id)] = {
            "held_out_from_background_training": True,
            "top1_business_id": top1_id,
            "top1_similarity": top1_similarity,
            "top2_business_id": top2_id,
            "top2_similarity": top2_similarity,
            "margin": margin,
        }
    return results


def _write_verified_png(path: Path, pixels: np.ndarray, business_id: int) -> None:
    Image.fromarray(pixels).save(
        path,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    expected = EXPECTED_OUTPUTS[business_id]
    if path.stat().st_size != expected["size"]:
        raise RuntimeError(f"generated reference {business_id} encoded size drifted")
    if _sha256_file(path) != expected["png_sha256"]:
        raise RuntimeError(f"generated reference {business_id} PNG SHA-256 drifted")
    with Image.open(path) as image:
        pixel_hash = _sha256_bytes(image.convert("RGB").tobytes())
    if pixel_hash != expected["rgb_pixel_sha256"]:
        raise RuntimeError(f"generated reference {business_id} pixel SHA-256 drifted")


def main() -> int:
    args = _parse_args()
    if PIL.__version__ != PINNED_PILLOW_VERSION or np.__version__ != PINNED_NUMPY_VERSION:
        raise RuntimeError("Pillow/NumPy versions do not match the frozen render toolchain")
    if _sha256_file(args.official_manifest_cache) != OFFICIAL_MANIFEST_CACHE_SHA256:
        raise RuntimeError("official PC manifest cache does not match 705100 revision 45")
    _validate_catalog(args.active_catalog)
    background_business_ids, rendered, official = _load_background_inputs(
        args.base_source_icons,
        args.crosswalk,
        args.official_art_root,
    )
    art = _extract_official_art(
        args.official_bundle,
        args.object_manager_root,
        OFFICIAL_ASSET,
    )
    art_339 = _extract_official_art(
        args.validation_bundle_339,
        args.object_manager_root,
        VALIDATION_ASSETS["img_general_pitem_3-339"],
    )
    art_340 = _extract_official_art(
        args.validation_bundle_340,
        args.object_manager_root,
        VALIDATION_ASSETS["img_general_pitem_3-340"],
    )
    background = _recover_background(
        rendered,
        official,
        expected_minimum_clear_count=MINIMUM_CLEAR_BACKGROUND_SAMPLES,
    )
    leave_one_out = _validate_leave_one_out(
        background,
        background_business_ids,
        args.base_source_icons,
        art_339,
        art_340,
    )
    normal = _compose_reference(background, art)
    upgraded = _apply_upgrade_marker(normal, args.base_source_icons)

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise RuntimeError("output directory must be empty")
    for source in _numeric_pngs(args.base_source_icons):
        shutil.copy2(source, args.output / source.name)
    _write_verified_png(args.output / "475.png", normal, 475)
    _write_verified_png(args.output / "476.png", upgraded, 476)
    final_paths = _numeric_pngs(args.output)
    if len(final_paths) != FINAL_REFERENCE_COUNT:
        raise RuntimeError("generated reference set does not contain exactly 426 IDs")

    provenance = {
        "schema_version": 1,
        "generator": "p_item_synthetic_rendered_extension_v1",
        "catalog": {
            "repository": "https://github.com/surisuririsu/gakumas-tools",
            "revision": ACTIVE_RIS_REVISION,
            "p_items_sha256": ACTIVE_CATALOG_SHA256,
            "included_business_ids": [475, 476],
            "stage_scope_contract": (
                "EntityBank selectable entities use modes=['stage']; "
                "coverage/generate-suite filters p.mode === 'stage'"
            ),
            "excluded_current_rows": [
                {
                    "business_id": 477,
                    "name": EXPECTED_CATALOG_ROWS[477]["name"],
                    "source_type": EXPECTED_CATALOG_ROWS[477]["sourceType"],
                    "mode": EXPECTED_CATALOG_ROWS[477]["mode"],
                    "reason": "produce-mode row is outside the upstream stage entity contract",
                }
            ],
        },
        "official_source": {
            "pc_application_version": 705100,
            "manifest_revision": 45,
            "manifest_cache_sha256": OFFICIAL_MANIFEST_CACHE_SHA256,
            **OFFICIAL_ASSET,
        },
        "decoder": {
            "project": "GkmasObjectManager",
            "revision": "7da94255b10e367bf83fadb805a0c5f643998de9",
            "license": "GPL-3.0",
            "runtime_only": True,
            "unitypy_version": PINNED_UNITYPY_VERSION,
            "pillow_version": PINNED_PILLOW_VERSION,
            "numpy_version": PINNED_NUMPY_VERSION,
        },
        "render_contract": {
            "base_reference_count": BASE_REFERENCE_COUNT,
            "base_source_icons_sha256": BASE_SOURCE_ICONS_SHA256,
            "crosswalk_sha256": CROSSWALK_SHA256,
            "background_sample_rule": "unupgraded SSR",
            "background_sample_count": BACKGROUND_SAMPLE_COUNT,
            "minimum_clear_background_samples": MINIMUM_CLEAR_BACKGROUND_SAMPLES,
            "resampling": "Pillow LANCZOS",
            "median_rounding": "numpy median then round-to-nearest-even",
            "target_size": list(TARGET_SIZE),
            "official_art_size": list(ART_SIZE),
            "official_art_offset": list(ART_OFFSET),
            "upgrade_marker_source_ids": list(MARKER_SOURCE_IDS),
            "upgrade_marker_box": list(MARKER_BOX),
            "upgrade_marker_difference_threshold": MARKER_DIFFERENCE_THRESHOLD,
            "upgrade_marker_pixel_count": MARKER_PIXEL_COUNT,
        },
        "generated": {
            str(business_id): {
                "name": EXPECTED_CATALOG_ROWS[business_id]["name"],
                "upgraded": EXPECTED_CATALOG_ROWS[business_id]["upgraded"],
                **expected,
            }
            for business_id, expected in EXPECTED_OUTPUTS.items()
        },
        "validation": {
            "historical_leave_one_out": leave_one_out,
            "historical_validation_assets": list(VALIDATION_ASSETS.values()),
            "generated_475_476_pair_correlation_64px": 0.95908997,
            "generated_475_476_theoretical_pair_margin": 0.04091,
            "full_gallery_64px": {
                "475": {
                    "top1_business_id": 475,
                    "top1_similarity": 1.0,
                    "top2_business_id": 476,
                    "top2_similarity": 0.96014803647995,
                    "margin": 0.03985196352005,
                },
                "476": {
                    "top1_business_id": 476,
                    "top1_similarity": 0.9999990463256836,
                    "top2_business_id": 475,
                    "top2_similarity": 0.96014803647995,
                    "margin": 0.03985100984573364,
                },
                "coarse_fine_guard": "fails_safe_to_full_gallery",
            },
            "real_475_476_jjc_sample": "PENDING",
        },
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
