from __future__ import annotations

import io
import json
import hashlib
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed rendered-domain P-item runtime gallery",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument(
        "--provisional-business-id",
        type=int,
        action="append",
        default=[],
        help="Business ID requiring runtime detail confirmation; repeat as needed",
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-scope", required=True)
    parser.add_argument(
        "--source-provenance",
        type=Path,
        help="Optional auditable extension-provenance JSON embedded in the manifest",
    )
    parser.add_argument("--evidence-id", default="task410-owner-pitem-r88")
    parser.add_argument("--jjc-reference-identity", default="CURRENT_VISIBLE_SUBSET")
    parser.add_argument(
        "--coarse-fine-validation",
        default="TOP2_DECISION_EQUIVALENT_100_OF_100_CANDIDATE24_GUARD12",
    )
    parser.add_argument(
        "--upgraded-marker-validation",
        default="PENDING_REAL_UPGRADED_SAMPLE",
    )
    return parser.parse_args()


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


def _encoded_digest(ids: list[int], encoded_images: list[bytes]) -> str:
    digest = hashlib.sha256()
    for business_id, encoded in zip(ids, encoded_images, strict=True):
        digest.update(str(business_id).encode("ascii"))
        digest.update(hashlib.sha256(encoded).digest())
    return digest.hexdigest().upper()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sanitized_png_bytes(path: Path) -> bytes:
    original = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if original is None or original.ndim != 3 or original.shape[2] != 3:
        raise RuntimeError(f"reference {path} is not a decodable BGR icon")
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            buffer = io.BytesIO()
            rgb.save(buffer, format="PNG", optimize=False, compress_level=9)
    except Exception as error:
        raise RuntimeError(f"reference {path} cannot be sanitized") from error
    encoded = buffer.getvalue()
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or not np.array_equal(original, decoded):
        raise RuntimeError(f"reference {path} changed pixels during metadata removal")
    with Image.open(io.BytesIO(encoded)) as sanitized:
        if sanitized.info or sanitized.getexif():
            raise RuntimeError(f"reference {path} still contains PNG metadata after sanitization")
    return encoded


def _validated_provisional_ids(
    provisional_ids: list[int], gallery_ids: list[int]
) -> list[int]:
    if (
        len(provisional_ids) != len(set(provisional_ids))
        or any(value < 1 for value in provisional_ids)
        or not set(provisional_ids).issubset(gallery_ids)
    ):
        raise RuntimeError("provisional P-item IDs must be unique members of the gallery")
    return sorted(provisional_ids)


def main() -> int:
    args = _parse_args()
    extension_provenance = None
    if args.source_provenance is not None:
        extension_provenance = json.loads(
            args.source_provenance.read_text(encoding="utf-8")
        )
        if not isinstance(extension_provenance, dict):
            raise RuntimeError("source provenance must be a JSON object")
    paths = sorted(
        (path for path in args.source.glob("*.png") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if len(paths) != args.expected_count:
        raise RuntimeError(
            f"expected {args.expected_count} numeric PNG references, found {len(paths)}"
        )
    ids: list[int] = []
    encoded_images: list[bytes] = []
    image_shapes: list[tuple[int, int, int]] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"reference {path} is not a decodable BGR icon")
        ids.append(int(path.stem))
        encoded_images.append(_sanitized_png_bytes(path))
        image_shapes.append(tuple(int(value) for value in image.shape))
    if len(ids) != len(set(ids)):
        raise RuntimeError("rendered P-item reference IDs are not unique")
    provisional_ids = _validated_provisional_ids(args.provisional_business_id, ids)

    args.output.mkdir(parents=True, exist_ok=True)
    gallery_path = args.output / "p_item_rendered_reference_gallery.npz"
    offsets = np.zeros(len(encoded_images) + 1, dtype=np.int64)
    for index, encoded in enumerate(encoded_images, start=1):
        offsets[index] = offsets[index - 1] + len(encoded)
    np.savez_compressed(
        gallery_path,
        p_item_ids=np.asarray(ids, dtype=np.int32),
        png_bytes=np.frombuffer(b"".join(encoded_images), dtype=np.uint8),
        png_offsets=offsets,
    )
    manifest = {
        "schema_version": 1,
        "recognizer": "p_item_rendered_reference_v1",
        "source": {
            "revision": args.source_revision,
            "scope": args.source_scope,
            "business_id_count": len(ids),
            "source_icons_sha256": _source_digest(paths),
            "sanitized_icons_sha256": _encoded_digest(ids, encoded_images),
            "evidence_id": args.evidence_id,
            "plan_ambiguous_business_ids": [406, 407, 408],
            "plan_to_business_id": {
                "sense": 406,
                "logic": 407,
                "anomaly": 408,
            },
        },
        "reference_gallery": {
            "path": gallery_path.name,
            "sha256": _sha256_file(gallery_path),
            "image_shape": "original_png_dimensions",
            "image_shape_count": len(set(image_shapes)),
            "color_order": "BGR",
            "business_id_count": len(ids),
            "provisional_business_ids": provisional_ids,
        },
        "runtime": {
            "source_color_order": "BGR",
            "stable_frame_count": 3,
            "content_generation_max_mean_abs_error": 0.75,
            "empty_foreground_maximum": 0.05,
            "nonempty_foreground_minimum": 0.15,
            "complete_opposite_half_minimum": 0.05,
            "identity_minimum_similarity": 0.65,
            "identity_minimum_margin": 0.01,
            "maximum_detail_fallbacks_per_member": 2,
            "identity_authority": "fixed_rendered_reference_coarse_fine_with_full_fallback",
            "embedding_role": "offline_diagnostic_only",
            "coarse_fine_profiles": [
                {
                    "minimum_capture_size": [720, 1280],
                    "canonical_slot_size": [64, 64],
                    "coarse_delta": -6,
                    "candidate_limit": 24,
                    "acceptance_rank_limit": 12,
                }
            ],
            "ambiguous_action": "bounded_detail_or_fail_closed",
        },
        "validation": {
            "jjc_reference_identity": args.jjc_reference_identity,
            "embedding_acceptance_calibration": "REJECTED_FOR_CURRENT_JJC_DOMAIN",
            "coarse_fine_720_calibration": args.coarse_fine_validation,
            "upgraded_marker_calibration": args.upgraded_marker_validation,
        },
        "build": {
            "deterministic_output": "NPZ_AND_CANONICAL_LF_JSON",
            "tool_contract": "task095-p-item-rendered-reference-gallery-v1",
        },
        "production_handoff": {
            "reason": ("fixed rendered-reference candidate requires lineage, container, and ranking evaluation before promotion"),
            "status": "PENDING_VALIDATION",
        },
    }
    if extension_provenance is not None:
        manifest["source"]["extension_provenance"] = extension_provenance
        if extension_provenance.get("schema_version") in {3, 4, 5, 6}:
            manifest["source"]["business_ids"] = ids
        if extension_provenance.get("schema_version") == 5:
            manifest["validation"] = {
                "jjc_reference_identity": "SCOPED_CALIBRATION_REQUIRES_PROMOTER_REPLAY",
                "embedding_acceptance_calibration": "REJECTED_FOR_CURRENT_JJC_DOMAIN",
                "coarse_fine_720_calibration": "SCOPED_CALIBRATION_REQUIRES_PROMOTER_REPLAY",
                "upgraded_marker_calibration": "GLOBAL_REAL_SAMPLE_CALIBRATION_PENDING",
            }
    (args.output / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
