"""Build compact card-face cost masks from fixed catalog and clean references."""

from __future__ import annotations

import json
import hashlib
import argparse
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--badge-gallery", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


def main() -> int:
    import cv2
    import numpy as np
    from arena_winrate.catalog import ArenaEntityCatalog
    from arena_winrate.card_cost import GenericCostReferenceGallery

    args = _arguments()
    catalog_dir = args.catalog_dir.resolve()
    cards = json.loads((catalog_dir / "skill_cards.json").read_text(encoding="utf-8"))
    customizations = json.loads(
        (catalog_dir / "customizations.json").read_text(encoding="utf-8")
    )
    catalog = ArenaEntityCatalog(cards, customizations)
    with np.load(args.badge_gallery.resolve(), allow_pickle=False) as payload:
        badge_ids = np.asarray(payload["business_ids"], dtype=np.int16)
        badge_green_masks = np.asarray(payload["overlap_masks"], dtype=np.uint8)
        reference_names = np.asarray(payload["reference_names"])

    roi_x, roi_y, roi_width, roi_height = GenericCostReferenceGallery.ROI

    def carrier(mask: object, *, source: str) -> object:
        normalized, _, _ = GenericCostReferenceGallery._anchored_carrier(mask)
        if normalized is None:
            raise ValueError(f"card-face cost carrier is missing from {source}")
        return normalized

    def majority(masks: list[object]) -> object:
        if not masks:
            raise ValueError("a card-face cost class has no fixed references")
        return (np.mean(np.stack(masks), axis=0) >= 0.5).astype(np.uint8)

    def red_mask(path: Path) -> object | None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode stamina-cost reference {path}")
        normalized = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
        normalized_carrier, _, _ = GenericCostReferenceGallery._anchored_carrier(
            GenericCostReferenceGallery._cost_color_mask(normalized, "stamina")
        )
        return normalized_carrier

    green_by_card: dict[int, list[object]] = {}
    green_by_value: dict[int, list[object]] = {}
    symbol_by_card: dict[int, list[object]] = {}
    symbol_by_kind_value: dict[tuple[str, int], list[object]] = {}
    used_symbol_paths: list[Path] = []
    for index, raw_card_id in enumerate(badge_ids):
        card_id = int(raw_card_id)
        descriptor = catalog.card_face_cost_descriptor(card_id)
        if descriptor is None or descriptor[0] != "cost" or descriptor[1] < 1:
            continue
        mask = carrier(
            badge_green_masks[index],
            source=f"badge gallery card {card_id} index {index}",
        )
        green_by_card.setdefault(card_id, []).append(mask)
        green_by_value.setdefault(descriptor[1], []).append(mask)
    for index, raw_card_id in enumerate(badge_ids):
        card_id = int(raw_card_id)
        descriptor = catalog.card_face_cost_descriptor(card_id)
        if descriptor is None or descriptor[0] != "cost" or descriptor[1] < 1:
            continue
        path = args.reference_root.resolve() / str(reference_names[index])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        normalized_image = cv2.resize(
            image,
            (96, 96),
            interpolation=cv2.INTER_AREA,
        )
        full_mask = badge_green_masks[index]
        _, _, carrier_box = GenericCostReferenceGallery._anchored_carrier(full_mask)
        if carrier_box is None:
            continue
        symbol = GenericCostReferenceGallery._symbol_descriptor(
            normalized_image,
            full_mask,
            carrier_box,
        )
        if symbol is None:
            continue
        symbol_by_card.setdefault(card_id, []).append(symbol)
        symbol_by_kind_value.setdefault(descriptor, []).append(symbol)
        used_symbol_paths.append(path)

    red_by_card: dict[int, list[object]] = {}
    red_by_value: dict[int, list[object]] = {}
    used_red_paths: list[Path] = []
    for path in sorted(args.reference_root.resolve().glob("*.png")):
        if not path.stem.isdigit():
            continue
        card_id = int(path.stem)
        descriptor = catalog.card_face_cost_descriptor(card_id)
        if descriptor is None or descriptor[0] != "stamina" or descriptor[1] < 1:
            continue
        mask = red_mask(path)
        if mask is None:
            continue
        red_by_card.setdefault(card_id, []).append(mask)
        red_by_value.setdefault(descriptor[1], []).append(mask)
        used_red_paths.append(path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        normalized_image = cv2.resize(
            image,
            (96, 96),
            interpolation=cv2.INTER_AREA,
        )
        full_mask = GenericCostReferenceGallery._cost_color_mask(
            normalized_image,
            "stamina",
        )
        _, _, carrier_box = GenericCostReferenceGallery._anchored_carrier(full_mask)
        if carrier_box is not None:
            symbol = GenericCostReferenceGallery._symbol_descriptor(
                normalized_image,
                full_mask,
                carrier_box,
            )
            if symbol is not None:
                symbol_by_card.setdefault(card_id, []).append(symbol)
                symbol_by_kind_value.setdefault(descriptor, []).append(symbol)
                used_symbol_paths.append(path)

    generic_card_ids: list[int] = []
    base_values: list[int] = []
    cost_kinds: list[str] = []
    base_masks: list[object] = []
    base_symbols: list[object] = []
    required_templates: set[tuple[str, int]] = set()
    for card_id in catalog.generic_cost_card_ids():
        descriptor = catalog.card_face_cost_descriptor(card_id)
        hypotheses = catalog.generic_cost_hypotheses(card_id)
        if descriptor is None or hypotheses is None:
            raise ValueError(f"generic-cost card {card_id} lacks a card-face descriptor")
        kind, base_value = descriptor
        references = (
            green_by_card.get(card_id, [])
            if kind == "cost"
            else red_by_card.get(card_id, [])
        )
        generic_card_ids.append(card_id)
        base_values.append(base_value)
        cost_kinds.append(kind)
        base_masks.append(majority(references))
        base_symbols.append(majority(symbol_by_card.get(card_id, [])))
        required_templates.add((kind, base_value))
        if hypotheses[1] > 0:
            required_templates.add((kind, hypotheses[1]))

    template_kinds: list[str] = []
    template_values: list[int] = []
    template_masks: list[object] = []
    template_symbols: list[object] = []
    for kind, value in sorted(required_templates):
        references = (
            green_by_value.get(value, [])
            if kind == "cost"
            else red_by_value.get(value, [])
        )
        template_kinds.append(kind)
        template_values.append(value)
        template_masks.append(majority(references))
        template_symbols.append(
            majority(symbol_by_kind_value.get((kind, value), []))
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = output_dir / "card_cost_reference.npz"
    np.savez_compressed(
        gallery_path,
        generic_card_ids=np.asarray(generic_card_ids, dtype=np.int16),
        base_values=np.asarray(base_values, dtype=np.int8),
        cost_kinds=np.asarray(cost_kinds),
        base_masks=np.stack(base_masks).astype(np.uint8),
        template_values=np.asarray(template_values, dtype=np.int8),
        template_kinds=np.asarray(template_kinds),
        template_masks=np.stack(template_masks).astype(np.uint8),
        base_symbols=np.stack(base_symbols).astype(np.uint8),
        template_symbols=np.stack(template_symbols).astype(np.uint8),
    )
    source_digest = hashlib.sha256()
    for path in (
        catalog_dir / "skill_cards.json",
        catalog_dir / "customizations.json",
        args.badge_gallery.resolve(),
        *sorted(set(used_red_paths + used_symbol_paths)),
    ):
        source_digest.update(str(path.name).encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(bytes.fromhex(_sha256(path)))
    manifest = {
        "schema_version": 1,
        "component": "task410_arena_card_face_cost_reference",
        "source": {
            "repository": "https://github.com/surisuririsu/gakumas-tools",
            "revision": args.source_revision,
            "license": "BSD-3-Clause",
            "source_set_sha256": source_digest.hexdigest().upper(),
        },
        "gallery": {
            "path": gallery_path.name,
            "sha256": _sha256(gallery_path),
            "generic_card_count": len(generic_card_ids),
            "template_count": len(template_values),
            "normalization_size": [96, 96],
            "roi": [roi_x, roi_y, roi_width, roi_height],
            "carrier_shape": [24, 24],
            "symbol_shape": [2, 24, 24],
        },
        "runtime": {
            "maximum_mask_error": 0.11,
            "minimum_class_margin": 0.01,
            "maximum_medium_error": 0.15,
            "minimum_medium_margin": 0.04,
            "maximum_high_error": 0.20,
            "maximum_cutout_error": 0.16,
            "minimum_cutout_margin": 0.04,
            "maximum_digit_error": 0.18,
            "minimum_digit_margin": 0.05,
            "rule_version": "arena-generic-cost-v8-id-independent-zero-topology",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
