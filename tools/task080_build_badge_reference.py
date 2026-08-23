"""Build compact clean-card references and offline green-mask audit data."""

from __future__ import annotations

import json
import hashlib
import argparse
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--arena-gallery", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


def main() -> int:
    import cv2
    import numpy as np

    args = _parse_args()
    reference_root = args.reference_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.arena_gallery.resolve(), allow_pickle=False) as payload:
        card_groups: dict[int, str] = {}
        for card_id, visual_group_id in zip(
            payload["card_ids"],
            payload["visual_group_ids"],
            strict=True,
        ):
            card_groups[int(str(card_id))] = str(visual_group_id)

    business_ids: list[int] = []
    visual_group_ids: list[str] = []
    names: list[str] = []
    top_coarse: list[object] = []
    green_masks: list[object] = []
    source_digest = hashlib.sha256()
    paths = sorted(
        (
            path
            for path in reference_root.glob("*.png")
            if path.stem.split("_", 1)[0].isdigit()
        ),
        key=lambda path: path.name,
    )
    for path in paths:
        business_id = int(path.stem.split("_", 1)[0])
        group = card_groups.get(business_id)
        if group is None:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode fixed reference: {path}")
        normalized = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(normalized.astype(np.int16))
        hsv_green = cv2.inRange(hsv, (30, 55, 45), (100, 255, 255)) > 0
        dominant = (g >= 65) & ((g - np.maximum(r, b)) >= 14)
        business_ids.append(business_id)
        visual_group_ids.append(group)
        names.append(path.name)
        top_coarse.append(
            cv2.resize(normalized[:48], (16, 8), interpolation=cv2.INTER_AREA)
        )
        green_masks.append((hsv_green | dominant).astype(np.uint8))
        source_digest.update(path.name.encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(bytes.fromhex(_sha256_file(path)))

    if not business_ids:
        raise ValueError("fixed reference directory yielded no mapped skill cards")
    gallery_path = output_dir / "badge_reference_gallery.npz"
    np.savez_compressed(
        gallery_path,
        business_ids=np.asarray(business_ids, dtype=np.int16),
        visual_group_ids=np.asarray(visual_group_ids),
        reference_names=np.asarray(names),
        top_coarse=np.stack(top_coarse).astype(np.uint8),
        green_masks=np.stack(green_masks).astype(np.uint8),
    )
    manifest = {
        "schema_version": 1,
        "component": "task410_arena_card_reference",
        "source": {
            "repository": "https://github.com/surisuririsu/gakumas-tools",
            "revision": args.source_revision,
            "license": "BSD-3-Clause",
            "reference_file_count": len(business_ids),
            "reference_set_sha256": source_digest.hexdigest().upper(),
        },
        "gallery": {
            "path": gallery_path.name,
            "sha256": _sha256_file(gallery_path),
            "business_card_id_count": len(set(business_ids)),
            "visual_group_count": len(set(visual_group_ids)),
            "normalization_size": [96, 96],
        },
        "runtime": {
            "maximum_coarse_error": 60.0,
            "maximum_zero_coarse_error": 65.0,
            "minimum_group_margin": 0.75,
            "minimum_high_error_group_margin": 3.0,
            "reference_rule_version": "task410-r10-card-identity-only-v1",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
