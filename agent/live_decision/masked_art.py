"""Prepared references for partially visible Live artwork.

Preparation is explicit and must precede a fresh observation. Retrieval never
builds a gallery or reads files inside an input authorization window.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import replace

import numpy as np
from PIL import Image

try:
    from agent.card_selection.model import ModelIntegrityError, sha256_file
    from agent.card_selection.embedding import EmbeddingGallery, EmbeddingCardRecognizer
except ModuleNotFoundError:
    from card_selection.model import ModelIntegrityError, sha256_file
    from card_selection.embedding import EmbeddingGallery, EmbeddingCardRecognizer

from .partial_card import LivePartialCardView, preprocess_live_card_view


def _source_reference_metadata(gallery):
    """Derive the identity boundary from the bound base gallery itself."""
    ids = gallery.card_ids
    enabled = getattr(gallery, "reference_enabled", None)
    enabled = np.ones(len(ids), dtype=np.bool_) if enabled is None else np.asarray(enabled)
    if enabled.dtype != np.bool_ or enabled.shape != (len(ids),) or not enabled.any():
        raise ModelIntegrityError("source gallery reference mask is invalid")
    metadata = {}
    disabled = set()
    for card_id, group, upgrade, active in zip(ids, gallery.visual_group_ids, gallery.upgrade_counts, enabled, strict=True):
        if not active:
            disabled.add(card_id)
            continue
        row = (group, upgrade)
        if card_id in metadata and metadata[card_id] != row:
            raise ModelIntegrityError("source gallery identity metadata is inconsistent")
        metadata[card_id] = row
    if disabled.intersection(metadata):
        raise ModelIntegrityError("source gallery mixes enabled and disabled rows for one card")
    return metadata, disabled


class PreparedMaskedArt:
    """Same-mask retrieval with complete enabled source-gallery coverage."""

    def __init__(self, recognizer, references, members, *, revision, exclusion_boxes=()):
        if not revision or set(references) != set(members):
            raise ValueError("masked references and members disagree")
        metadata, disabled = _source_reference_metadata(recognizer.gallery)
        if {cid for ids in members.values() for cid in ids} != set(metadata):
            raise ValueError("masked references must cover exactly the enabled source gallery IDs")
        for asset, ids in members.items():
            if not ids or len({metadata[cid][0] for cid in ids}) != 1:
                raise ValueError("one artwork must map to exactly one visual group")
            array = np.asarray(references[asset])
            if array.shape != (180, 180, 3) or array.dtype != np.uint8:
                raise ValueError("masked reference must be canonical 180x180 RGB")
        self.recognizer = recognizer
        self.references = {key: value.copy() for key, value in references.items()}
        self.members = {key: tuple(value) for key, value in members.items()}
        self.metadata = metadata
        self.revision = revision
        self._schema_version = 2 if disabled else 1
        # A bank owns one UI exclusion profile. Query and references must
        # consume exactly the same profile, independent of the card identity.
        self._exclusion_boxes = tuple(tuple(box) for box in exclusion_boxes)
        for box in self.exclusion_boxes:
            if (len(box) != 4 or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError("masked reference exclusion is invalid")
        self._prepared = {}

    @property
    def exclusion_boxes(self):
        return self._exclusion_boxes

    @classmethod
    def load(cls, recognizer, dataset_root, *, additional_dataset_roots=(), exclusion_boxes=()):
        members = {}
        references = {}
        revisions = []
        identities = {}
        digests = {}
        for dataset in (dataset_root, *additional_dataset_roots):
            root = Path(dataset)
            manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
            crosswalk_path = root / "crosswalk.json"
            if sha256_file(crosswalk_path) != manifest["mapping"]["crosswalk_sha256"].upper():
                raise ModelIntegrityError("masked dataset crosswalk SHA-256 mismatch")
            revisions.append(manifest["dataset_id"])
            crosswalk = json.loads(crosswalk_path.read_text(encoding="utf-8"))
            records = {row["asset_name"]: row for row in manifest["art"]["images"]}
            for row in crosswalk:
                card_id = str(row["business_id"])
                identity = (row["internal_id"], row["upgrade_count"], frozenset(row["asset_variants"]))
                if identities.setdefault(card_id, identity) != identity:
                    raise ModelIntegrityError("masked datasets disagree on a card identity")
                if not identity[2]:
                    raise ModelIntegrityError("masked dataset identity has no artwork")
                for asset in row["asset_variants"]:
                    if asset not in records:
                        raise ModelIntegrityError("masked reference is missing from pinned manifest")
                    record = records[asset]
                    path = root / "art" / Path(record["path"]).name
                    digest = record["sha256"].upper()
                    if digests.setdefault(asset, digest) != digest:
                        raise ModelIntegrityError("masked datasets disagree on an artwork digest")
                    if asset not in references:
                        if sha256_file(path) != digest:
                            raise ModelIntegrityError("masked reference SHA-256 mismatch")
                        with Image.open(path) as source:
                            references[asset] = np.asarray(source.convert("RGB").resize((180, 180), Image.Resampling.BILINEAR))
                    members.setdefault(asset, set()).add(card_id)
        return cls(recognizer, references, members, revision=" + ".join(revisions),
                   exclusion_boxes=exclusion_boxes)

    @staticmethod
    def mask_key(view):
        # Quantize inward so the common mask never includes unobserved pixels.
        # The grid is independent of card IDs, slots and observed screenshots.
        width, height = view.canvas_size
        ox, oy = view.canvas_offset
        visible_width, visible_height = view.visible_box[2:]
        if width != height or oy != 0 or visible_height != height:
            raise ValueError("masked Live art requires a full-height square canvas")
        if ox < 0 or visible_width <= 0 or ox + visible_width > width:
            raise ValueError("visible artwork exceeds its canvas")
        left = next(i for i in range(33) if int(np.ceil(i * width / 32)) >= ox)
        right = max(i for i in range(33) if int(np.floor(i * width / 32)) <= ox + visible_width)
        if not 0 <= left < right <= 32 or right - left < 8:
            raise ValueError("insufficient visible artwork for a common mask")
        return left, right

    @staticmethod
    def common_view(view, key):
        left, right = key
        width = view.canvas_size[0]
        start = int(np.ceil(left * width / 32))
        stop = int(np.floor(right * width / 32))
        source_left = view.visible_box[0] - view.canvas_offset[0]
        return replace(
            view,
            visible_box=(source_left + start, view.visible_box[1], stop - start, view.visible_box[3]),
            canvas_offset=(start, 0),
            visible_ratio=(stop - start) / width,
        )

    def prepare(self, views):
        """Build all requested masks before acquiring a new input snapshot."""
        keys = {self.mask_key(view) for view in views}
        if len(keys | self._prepared.keys()) > 64:
            raise ValueError("masked reference preparation budget exceeded")
        for key in sorted(keys - self._prepared.keys()):
            canonical = LivePartialCardView(0, (0, 0, 180, 180), (0, 0, 180, 180), (180, 180), 1.0, "full", "masked-art-v1")
            view = self.common_view(canonical, key)
            vectors, names, ids, groups, upgrades = [], [], [], [], []
            for asset, source in self.references.items():
                tensor = preprocess_live_card_view(source, view, source_color_order="RGB",
                                                   exclusion_boxes=self.exclusion_boxes)
                vector = self.recognizer.embedder.embed_tensor(tensor)
                for cid in sorted(self.members[asset], key=int):
                    vectors.append(vector)
                    names.append(f"{cid}_{asset}")
                    ids.append(cid)
                    group, upgrade = self.metadata[cid]
                    groups.append(group)
                    upgrades.append(upgrade)
            gallery = EmbeddingGallery(
                np.stack(vectors),
                names,
                ids,
                visual_group_ids=groups,
                upgrade_counts=upgrades,
                model_sha256=self.recognizer.model_sha256,
                gallery_sha256=(f"prepared:{self.revision}:{key[0]}:{key[1]}"
                                + (f":{self.exclusion_boxes}" if self.exclusion_boxes else "")),
            )
            self._prepared[key] = EmbeddingCardRecognizer(
                self.recognizer.embedder,
                gallery,
                classes_sha256=self.recognizer.classes_sha256,
                upgrade_maximum_distance=self.recognizer.upgrade_maximum_distance,
                upgrade_minimum_margin=self.recognizer.upgrade_minimum_margin,
            )

    def classify(self, image, view, candidate, **kwargs):
        key = self.mask_key(view)
        if key not in self._prepared:
            raise ValueError("masked reference has not been prepared before observation")
        tensor = preprocess_live_card_view(
            image,
            self.common_view(view, key),
            source_color_order=self.recognizer.embedder.source_color_order,
            exclusion_boxes=self.exclusion_boxes,
        )
        return self._prepared[key].classify_embedding(
            self.recognizer.embedder.embed_tensor(tensor),
            candidate,
            resolve_upgrade_state=False,
            **kwargs,
        )

    def export(self, output_root):
        """Freeze prepared vectors offline; ship no reference screenshots."""
        if not self._prepared:
            raise ValueError("no masked references have been prepared")
        if self.exclusion_boxes != ((0.0, 0.0, 0.70, 0.40),):
            raise ValueError("only the calibrated HIF exclusion profile can be frozen")
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._prepared)
        first = self._prepared[keys[0]].gallery
        metadata_names = ("class_names", "card_ids", "visual_group_ids", "upgrade_counts")
        payload = {name: np.asarray(getattr(first, name)) for name in metadata_names}
        for key in keys:
            gallery = self._prepared[key].gallery
            if any(tuple(getattr(gallery, name)) != tuple(getattr(first, name)) for name in metadata_names):
                raise ValueError("masked gallery identities differ across profiles")
            payload[f"embeddings_{key[0]}_{key[1]}"] = gallery.embeddings
        gallery_path = root / "gallery.npz"
        np.savez_compressed(gallery_path, **payload)
        manifest = {
            "schema_version": self._schema_version,
            "revision": "hif-card-art-exclusions-v1",
            "source_dataset": self.revision,
            "source_model_sha256": self.recognizer.model_sha256,
            "source_gallery_sha256": self.recognizer.gallery_sha256,
            "source_classes_sha256": self.recognizer.classes_sha256,
            "gallery_sha256": sha256_file(gallery_path),
            "mask_keys": keys,
            "exclusion_boxes": self.exclusion_boxes,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest


class FrozenMaskedArt:
    """Load an offline-built bundle before capture; never prepare during input."""

    mask_key = staticmethod(PreparedMaskedArt.mask_key)
    common_view = staticmethod(PreparedMaskedArt.common_view)

    def __init__(self, recognizer, prepared, exclusions, gallery_sha256):
        self.recognizer = recognizer
        self._prepared = prepared
        self._exclusion_boxes = exclusions
        self.gallery_sha256 = gallery_sha256

    @property
    def exclusion_boxes(self):
        return self._exclusion_boxes

    def classify(self, image, view, candidate, **kwargs):
        # The HIF bundle has its own validated floor. Debug thresholds may
        # tighten it, but cannot restore the legacy 0.70 / zero-margin gate.
        for name, minimum in (("acceptance_threshold", .85), ("min_margin", .05)):
            value = kwargs.get(name)
            if value is None:
                value = minimum
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("masked gallery threshold must be finite")
            kwargs[name] = max(minimum, value)
        return PreparedMaskedArt.classify(self, image, view, candidate, **kwargs)

    @classmethod
    def load(cls, recognizer, root):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        schema = manifest.get("schema_version")
        if type(schema) is not int or schema not in (1, 2) or manifest.get("revision") != "hif-card-art-exclusions-v1":
            raise ModelIntegrityError("unsupported masked gallery revision")
        for name in ("model_sha256", "gallery_sha256", "classes_sha256"):
            if manifest.get(f"source_{name}") != getattr(recognizer, name):
                raise ModelIntegrityError("masked gallery source differs from the current recognizer")
        exclusions = tuple(tuple(box) for box in manifest.get("exclusion_boxes", ()))
        if exclusions != ((0.0, 0.0, 0.70, 0.40),):
            raise ModelIntegrityError("masked gallery UI exclusion profile is uncalibrated")
        gallery_path = root / "gallery.npz"
        digest = sha256_file(gallery_path)
        if digest != manifest.get("gallery_sha256"):
            raise ModelIntegrityError("masked gallery SHA-256 mismatch")
        keys = manifest.get("mask_keys")
        if (not isinstance(keys, list) or not keys or len(keys) > 64
                or any(not isinstance(key, list) or len(key) != 2
                       or any(type(v) is not int for v in key)
                       or not 0 <= key[0] < key[1] <= 32 or key[1] - key[0] < 8 for key in keys)
                or len({tuple(key) for key in keys}) != len(keys)):
            raise ModelIntegrityError("masked gallery keys are invalid")
        expected, disabled = _source_reference_metadata(recognizer.gallery)
        if schema == 1 and disabled:
            raise ModelIntegrityError("legacy masked gallery cannot bind a source with disabled references")
        prepared = {}
        with np.load(gallery_path, allow_pickle=False) as payload:
            for left, right in keys:
                gallery = EmbeddingGallery(
                    payload[f"embeddings_{left}_{right}"], payload["class_names"], payload["card_ids"],
                    visual_group_ids=payload["visual_group_ids"], upgrade_counts=payload["upgrade_counts"],
                    model_sha256=recognizer.model_sha256, gallery_sha256=digest,
                )
                if (set(gallery.card_ids) != set(expected)
                        or any(expected.get(cid) != (group, upgrade)
                               for cid, group, upgrade in zip(gallery.card_ids, gallery.visual_group_ids,
                                                              gallery.upgrade_counts, strict=True))):
                    raise ModelIntegrityError("masked gallery identity coverage differs from the source")
                prepared[left, right] = EmbeddingCardRecognizer(
                    recognizer.embedder, gallery, classes_sha256=recognizer.classes_sha256,
                    upgrade_maximum_distance=recognizer.upgrade_maximum_distance,
                    upgrade_minimum_margin=recognizer.upgrade_minimum_margin,
                )
        return cls(recognizer, prepared, exclusions, digest)
