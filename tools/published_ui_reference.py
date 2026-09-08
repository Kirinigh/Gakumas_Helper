"""Replay frozen, third-party published UI crops without asserting game capture origin."""

from __future__ import annotations

import io
import re
import json
import hashlib
from typing import Any
from pathlib import Path
from datetime import date
from urllib.parse import urlsplit
from collections.abc import Mapping, Sequence

from PIL import Image

SOURCE_TYPE = "third_party_published_ui_crops"
REFERENCE_KEYS = frozenset(
    {
        "kind",
        "business_id",
        "name",
        "upgraded",
        "publisher",
        "page_url",
        "original_url",
        "original_file",
        "original_sha256",
        "original_size",
        "original_dimensions",
        "frozen_at",
        "capture_provenance",
        "crop_box",
        "resampling",
        "dimensions",
        "png_sha256",
        "rgb_pixel_sha256",
        "size",
    }
)
SHA256_PATTERN = re.compile(r"[0-9A-Fa-f]{64}")


class PublishedUiReferenceError(RuntimeError):
    """A published UI source record or its reproducible image is invalid."""


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _validate_reference(reference: Mapping[str, Any], *, component: str) -> None:
    if not isinstance(reference, Mapping) or set(reference) != REFERENCE_KEYS:
        raise PublishedUiReferenceError("published UI reference fields are invalid")
    if reference["kind"] != component or component not in {"p_item", "skill_card"}:
        raise PublishedUiReferenceError("published UI reference kind is invalid")
    if not _positive_integer(reference["business_id"]) or not isinstance(reference["upgraded"], bool):
        raise PublishedUiReferenceError("published UI reference identity is invalid")
    if any(not isinstance(reference[key], str) or not reference[key].strip() for key in ("name", "publisher")):
        raise PublishedUiReferenceError("published UI reference name or publisher is invalid")
    for key in ("page_url", "original_url"):
        try:
            url = urlsplit(reference[key])
            valid = url.scheme == "https" and bool(url.hostname) and not url.username and not url.password
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise PublishedUiReferenceError(f"published UI reference {key} is invalid")
    filename = reference["original_file"]
    if not isinstance(filename, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", filename) is None:
        raise PublishedUiReferenceError("published UI original_file must be a local basename")
    for key in ("original_sha256", "png_sha256", "rgb_pixel_sha256"):
        if not isinstance(reference[key], str) or SHA256_PATTERN.fullmatch(reference[key]) is None:
            raise PublishedUiReferenceError(f"published UI reference {key} is invalid")
    if any(not _positive_integer(reference[key]) for key in ("original_size", "size")):
        raise PublishedUiReferenceError("published UI reference byte size is invalid")
    dimensions = reference["original_dimensions"]
    if not isinstance(dimensions, list) or len(dimensions) != 2 or not all(map(_positive_integer, dimensions)):
        raise PublishedUiReferenceError("published UI original dimensions are invalid")
    box = reference["crop_box"]
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(not isinstance(value, int) or isinstance(value, bool) for value in box)
        or not (0 <= box[0] < box[2] <= dimensions[0] and 0 <= box[1] < box[3] <= dimensions[1])
    ):
        raise PublishedUiReferenceError("published UI crop_box is outside the original image")
    if (
        reference["dimensions"] != [130, 130]
        or not all(type(value) is int for value in reference["dimensions"])
        or reference["resampling"] != "PILLOW_BICUBIC"
    ):
        raise PublishedUiReferenceError("published UI output dimensions or resampling are invalid")
    try:
        frozen_at = reference["frozen_at"]
        if not isinstance(frozen_at, str) or date.fromisoformat(frozen_at).isoformat() != frozen_at:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise PublishedUiReferenceError("published UI frozen_at must be an ISO date") from error
    if reference["capture_provenance"] != "UNKNOWN":
        raise PublishedUiReferenceError("published UI capture provenance must remain UNKNOWN")


def validate_published_ui_references(
    references: Mapping[str, Any],
    *,
    expected_ids: Sequence[int],
    catalog_rows: Sequence[Mapping[str, Any]] | Mapping[str | int, Mapping[str, Any]] | None = None,
    component: str = "p_item",
) -> None:
    """Validate public records; optionally bind names and upgrade state to the catalog."""
    if not expected_ids or any(not _positive_integer(value) for value in expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise PublishedUiReferenceError("published UI expected business IDs are invalid")
    if not isinstance(references, Mapping) or set(references) != {str(value) for value in expected_ids}:
        raise PublishedUiReferenceError("published UI reference IDs differ from the declared changes")
    rows: dict[str, Mapping[str, Any]] = {}
    if catalog_rows is not None:
        entries = catalog_rows.items() if isinstance(catalog_rows, Mapping) else ((row.get("id"), row) for row in catalog_rows)
        for business_id, row in entries:
            if str(business_id) in rows or not isinstance(row, Mapping):
                raise PublishedUiReferenceError("published UI catalog rows are invalid or duplicated")
            rows[str(business_id)] = row
    for key, reference in references.items():
        _validate_reference(reference, component=component)
        if str(reference["business_id"]) != key:
            raise PublishedUiReferenceError("published UI reference business_id differs from its key")
        if catalog_rows is None:
            continue
        row = rows.get(key)
        if row is None or not isinstance(row.get("name"), str):
            raise PublishedUiReferenceError(f"published UI reference {key} has no catalog identity")
        catalog_name = row["name"]
        has_marker = catalog_name.endswith(("+", "＋"))
        upgraded = row.get("upgraded", has_marker)
        if not isinstance(upgraded, bool) or (has_marker and not upgraded):
            raise PublishedUiReferenceError(f"published UI reference {key} catalog upgrade state is invalid")
        name = catalog_name[:-1] if has_marker else catalog_name
        if reference["name"] != name or reference["upgraded"] != upgraded:
            raise PublishedUiReferenceError(f"published UI reference {key} name or upgrade state differs from catalog")


def validate_published_ui_reference_png(reference: Mapping[str, Any], png_bytes: bytes) -> None:
    """Validate the frozen output without requiring original image files at Release."""
    _validate_reference(reference, component=reference.get("kind", ""))
    if len(png_bytes) != reference["size"] or _sha256(png_bytes) != reference["png_sha256"].upper():
        raise PublishedUiReferenceError("published UI PNG byte size or SHA-256 differs from the record")
    try:
        with Image.open(io.BytesIO(png_bytes)) as image:
            if image.format != "PNG" or image.mode != "RGB" or list(image.size) != reference["dimensions"] or image.info or image.getexif():
                raise PublishedUiReferenceError("published UI output must be a metadata-free RGB PNG")
            if _sha256(image.tobytes()) != reference["rgb_pixel_sha256"].upper():
                raise PublishedUiReferenceError("published UI output RGB pixels differ from the record")
    except (OSError, ValueError) as error:
        raise PublishedUiReferenceError("published UI output PNG cannot be decoded") from error


def replay_published_ui_reference(reference: Mapping[str, Any], original_root: Path) -> bytes:
    """Reproduce a half-open crop and fixed Pillow bicubic resize from frozen bytes."""
    _validate_reference(reference, component=reference.get("kind", ""))
    root = Path(original_root).resolve()
    path = (root / reference["original_file"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise PublishedUiReferenceError("published UI original is unavailable or outside its root")
    raw = path.read_bytes()
    if len(raw) != reference["original_size"] or _sha256(raw) != reference["original_sha256"].upper():
        raise PublishedUiReferenceError("published UI original byte size or SHA-256 differs from the record")
    try:
        with Image.open(io.BytesIO(raw)) as original:
            if list(original.size) != reference["original_dimensions"]:
                raise PublishedUiReferenceError("published UI original dimensions differ from the record")
            cropped = original.convert("RGB").crop(tuple(reference["crop_box"]))
            resized = cropped.resize(tuple(reference["dimensions"]), resample=Image.Resampling.BICUBIC)
            clean = Image.frombytes("RGB", resized.size, resized.tobytes())
            output = io.BytesIO()
            clean.save(output, format="PNG", optimize=False, compress_level=9)
    except (OSError, ValueError) as error:
        raise PublishedUiReferenceError("published UI original cannot be replayed") from error
    encoded = output.getvalue()
    validate_published_ui_reference_png(reference, encoded)
    return encoded


def validate_published_ui_source_manifest(
    manifest_path: Path,
    *,
    original_root: Path,
    output_root: Path,
    catalog_rows: Sequence[Mapping[str, Any]] | Mapping[str | int, Mapping[str, Any]],
    component: str,
    expected_ids: Sequence[int],
) -> dict[str, Any]:
    """Validate a source manifest, frozen originals and the exact resulting PNG files."""
    try:
        manifest = json.loads(Path(manifest_path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublishedUiReferenceError("published UI source manifest cannot be read") from error
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "source_type", "references"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["source_type"] != SOURCE_TYPE
    ):
        raise PublishedUiReferenceError("published UI source manifest contract is invalid")
    validate_published_ui_references(manifest["references"], expected_ids=expected_ids, catalog_rows=catalog_rows, component=component)
    output_directory = Path(output_root).resolve()
    for key, reference in manifest["references"].items():
        replayed = replay_published_ui_reference(reference, original_root)
        output_path = (output_directory / f"{key}.png").resolve()
        if not output_path.is_relative_to(output_directory) or not output_path.is_file() or output_path.read_bytes() != replayed:
            raise PublishedUiReferenceError(f"published UI output {key}.png differs from the original crop replay")
    return manifest
