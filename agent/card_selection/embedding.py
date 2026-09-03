from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping

from .model import ModelIntegrityError, sha256_file
from .types import UNKNOWN, CandidateBox, CardPrediction

VISUAL_AMBIGUITY_GROUPS = {
    frozenset(("789", "791", "793")): {
        "sense": "789",
        "logic": "791",
        "anomaly": "793",
    },
    frozenset(("790", "792", "794")): {
        "sense": "790",
        "logic": "792",
        "anomaly": "794",
    },
}
VALID_PLAN_TYPES = frozenset(("sense", "logic", "anomaly"))
CARD_ART_EXCLUSION_BOXES = (
    (0.00, 0.00, 0.38, 0.28),
    (0.70, 0.00, 1.00, 0.36),
    (0.00, 0.64, 0.32, 1.00),
    (0.64, 0.58, 1.00, 1.00),
)
UPGRADE_MARKER_BOX = (0.75, 0.36, 0.98, 0.64)


def _declared_business_card_ids(source: Any) -> frozenset[str]:
    """Expand one manifest's exact business-card coverage declaration."""

    if not isinstance(source, Mapping):
        raise ModelIntegrityError("embedding source metadata is missing")

    declared: set[int] | None = None
    raw_ids = source.get("business_card_ids")
    if raw_ids is not None:
        if not isinstance(raw_ids, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in raw_ids
        ):
            raise ModelIntegrityError("embedding business-card ID declaration is invalid")
        if len(raw_ids) != len(set(raw_ids)):
            raise ModelIntegrityError("embedding business-card ID declaration contains duplicates")
        declared = set(raw_ids)

    raw_ranges = source.get("business_card_id_ranges")
    if raw_ranges is not None:
        if not isinstance(raw_ranges, list):
            raise ModelIntegrityError("embedding business-card ID ranges are invalid")
        expanded: set[int] = set()
        for item in raw_ranges:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in item)
                or item[0] < 1
                or item[1] < item[0]
            ):
                raise ModelIntegrityError("embedding business-card ID ranges are invalid")
            values = set(range(item[0], item[1] + 1))
            if expanded.intersection(values):
                raise ModelIntegrityError("embedding business-card ID ranges overlap")
            expanded.update(values)
        if declared is not None and declared != expanded:
            raise ModelIntegrityError("embedding business-card ID declarations disagree")
        declared = expanded

    minimum = source.get("business_card_id_min")
    maximum = source.get("business_card_id_max")
    if (minimum is None) != (maximum is None):
        raise ModelIntegrityError("embedding business-card ID range is incomplete")
    if minimum is not None:
        if (
            isinstance(minimum, bool)
            or isinstance(maximum, bool)
            or not isinstance(minimum, int)
            or not isinstance(maximum, int)
            or minimum < 1
            or maximum < minimum
        ):
            raise ModelIntegrityError("embedding business-card ID range is invalid")
        contiguous = set(range(minimum, maximum + 1))
        if declared is None:
            declared = contiguous
        elif declared != contiguous:
            raise ModelIntegrityError("embedding business-card ID range disagrees with the exact set")

    if not declared:
        raise ModelIntegrityError("embedding manifest has no exact business-card ID set")
    count = source.get("business_card_id_count")
    if count is not None and (
        isinstance(count, bool) or not isinstance(count, int) or count != len(declared)
    ):
        raise ModelIntegrityError("embedding business-card ID count disagrees with the exact set")
    return frozenset(str(value) for value in declared)


def focus_card_art(image: Any) -> Any:
    """Exclude score, cost and side-function UI from the embedding image."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError("Pillow is required for card-art preprocessing") from error

    source = image if isinstance(image, Image.Image) else Image.fromarray(image)
    focused = source.convert("RGB")
    draw = ImageDraw.Draw(focused)
    width, height = focused.size
    for left, top, right, bottom in CARD_ART_EXCLUSION_BOXES:
        draw.rectangle(
            (round(left * width), round(top * height), round(right * width), round(bottom * height)),
            fill=(255, 255, 255),
        )
    return focused


def extract_upgrade_marker(image: Any) -> Any:
    """Return the isolated right-side upgrade marker as a stable RGB tensor."""

    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("numpy and Pillow are required for upgrade-state preprocessing") from error

    source = image if isinstance(image, Image.Image) else Image.fromarray(image)
    rgb = source.convert("RGB")
    width, height = rgb.size
    left, top, right, bottom = UPGRADE_MARKER_BOX
    marker = rgb.crop(
        (round(left * width), round(top * height), round(right * width), round(bottom * height))
    ).resize((24, 24), Image.Resampling.BILINEAR)
    return np.asarray(marker, dtype=np.float32) / 255.0

@dataclass(frozen=True)
class EmbeddingHit:
    """One business-card result from cosine nearest-neighbour retrieval."""

    card_id: str
    class_name: str
    similarity: float
    source_index: int
    visual_group_id: str = ""
    candidate_card_ids: tuple[str, ...] = ()
    candidate_upgrade_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class ResolvedEmbeddingHit:
    """Top-1 business ID after deterministic plan-type ambiguity resolution."""

    card_id: str
    similarity: float
    source_card_id: str
    plan: str
    ambiguity_resolved: bool
    upgraded: bool | None = None
    upgrade_resolved: bool = False


def resolve_card_identity(hits: Any, *, plan: str, upgraded: bool | None) -> ResolvedEmbeddingHit:
    """Resolve one artwork group to an exact business ID after retrieval."""

    candidates = tuple(hits)
    if not candidates:
        raise ValueError("at least one embedding hit is required")
    top = candidates[0]
    card_ids = top.candidate_card_ids or (top.card_id,)
    upgrade_counts = top.candidate_upgrade_counts or tuple(0 for _ in card_ids)
    if len(card_ids) != len(upgrade_counts):
        raise ValueError("visual-group card IDs and upgrade states are inconsistent")
    if len(card_ids) == 1:
        return ResolvedEmbeddingHit(
            card_id=card_ids[0],
            similarity=top.similarity,
            source_card_id=top.card_id,
            plan=str(plan).strip().lower(),
            ambiguity_resolved=False,
        )
    if upgraded is None:
        raise ValueError("reliable upgrade state is required for a multi-ID artwork group")
    wanted_upgrade = 1 if upgraded else 0
    eligible = [card_id for card_id, count in zip(card_ids, upgrade_counts, strict=True) if count == wanted_upgrade]
    normalised_plan = str(plan).strip().lower()
    if set(card_ids) == {"789", "790", "791", "792", "793", "794"}:
        if normalised_plan not in VALID_PLAN_TYPES:
            raise ValueError("reliable plan type is required for the April-Fools artwork group")
        target_id = {
            ("sense", 0): "789",
            ("sense", 1): "790",
            ("logic", 0): "791",
            ("logic", 1): "792",
            ("anomaly", 0): "793",
            ("anomaly", 1): "794",
        }[(normalised_plan, wanted_upgrade)]
    elif len(eligible) == 1:
        target_id = eligible[0]
    else:
        raise ValueError("upgrade state does not resolve the retrieved artwork group uniquely")
    return ResolvedEmbeddingHit(
        card_id=target_id,
        similarity=top.similarity,
        source_card_id=top.card_id,
        plan=normalised_plan,
        ambiguity_resolved=len(card_ids) > 2,
        upgraded=upgraded,
        upgrade_resolved=True,
    )


def resolve_visual_ambiguity(hits: Any, plan: str) -> ResolvedEmbeddingHit:
    """Resolve only the two confirmed April-Fools visual-identity groups."""

    candidates = tuple(hits)
    if not candidates:
        raise ValueError("at least one embedding hit is required")
    normalised_plan = str(plan).strip().lower()
    top = candidates[0]
    for group, plan_mapping in VISUAL_AMBIGUITY_GROUPS.items():
        if top.card_id not in group:
            continue
        present = {hit.card_id for hit in candidates if hit.card_id in group}
        if present != set(group):
            raise ValueError("confirmed visual ambiguity group is incomplete in Top-K results")
        if normalised_plan not in VALID_PLAN_TYPES:
            raise ValueError("reliable plan type is required for visual ambiguity resolution")
        target_id = plan_mapping[normalised_plan]
        target = next(hit for hit in candidates if hit.card_id == target_id)
        return ResolvedEmbeddingHit(
            card_id=target_id,
            similarity=target.similarity,
            source_card_id=top.card_id,
            plan=normalised_plan,
            ambiguity_resolved=True,
        )
    return ResolvedEmbeddingHit(
        card_id=top.card_id,
        similarity=top.similarity,
        source_card_id=top.card_id,
        plan=normalised_plan,
        ambiguity_resolved=False,
    )


class EmbeddingGallery:
    """Validated, immutable embedding gallery for evaluation and production retrieval."""

    EMBEDDING_DIM = 128

    def __init__(
        self,
        embeddings: Any,
        class_names: Any,
        card_ids: Any,
        *,
        model_sha256: str,
        gallery_sha256: str = "IN_MEMORY",
        visual_group_ids: Any | None = None,
        upgrade_counts: Any | None = None,
        upgrade_markers: Any | None = None,
    ) -> None:
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("numpy is required for embedding retrieval") from error

        vectors = np.asarray(embeddings, dtype=np.float32)
        names = tuple(str(value) for value in class_names)
        ids = tuple(str(value) for value in card_ids)
        groups = tuple(str(value) for value in visual_group_ids) if visual_group_ids is not None else ids
        upgrades = (
            tuple(int(value) for value in upgrade_counts)
            if upgrade_counts is not None
            else tuple(0 for _ in ids)
        )
        markers = None if upgrade_markers is None else np.asarray(upgrade_markers, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.EMBEDDING_DIM:
            raise ModelIntegrityError("gallery must be an Nx128 matrix")
        if (
            vectors.shape[0] == 0
            or vectors.shape[0] != len(names)
            or len(names) != len(ids)
            or len(ids) != len(groups)
            or len(groups) != len(upgrades)
        ):
            raise ModelIntegrityError("gallery arrays have inconsistent lengths")
        if len(set(names)) != len(names):
            raise ModelIntegrityError("gallery class names must be unique")
        if not np.isfinite(vectors).all():
            raise ModelIntegrityError("gallery contains non-finite embeddings")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 1e-8):
            raise ModelIntegrityError("gallery contains a zero embedding")
        normalised = vectors / norms[:, None]
        if float(np.max(np.abs(norms - 1.0))) > 1e-3:
            raise ModelIntegrityError("gallery embeddings must be L2-normalised")
        for class_name, card_id in zip(names, ids, strict=True):
            expected_id = class_name.split("_", 1)[0]
            if not card_id.isdigit() or card_id != expected_id:
                raise ModelIntegrityError(f"invalid business-card mapping: {class_name} -> {card_id}")
        if any(not group for group in groups) or any(value not in {0, 1} for value in upgrades):
            raise ModelIntegrityError("gallery visual-group metadata is invalid")
        if markers is not None and (markers.shape != (len(ids), 24, 24, 3) or not np.isfinite(markers).all()):
            raise ModelIntegrityError("gallery upgrade-marker templates are invalid")
        self.embeddings = np.ascontiguousarray(normalised, dtype=np.float32)
        self.class_names = names
        self.card_ids = ids
        self.visual_group_ids = groups
        self.upgrade_counts = upgrades
        self.upgrade_markers = markers
        self.model_sha256 = model_sha256.upper()
        self.gallery_sha256 = gallery_sha256.upper()

    @classmethod
    def load(cls, gallery_path: str | Path, manifest_path: str | Path) -> EmbeddingGallery:
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("numpy is required for embedding retrieval") from error

        gallery_file = Path(gallery_path)
        manifest_file = Path(manifest_path)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
        if manifest.get("schema_version") != 1:
            raise ModelIntegrityError("unsupported embedding manifest schema")
        gallery_info = manifest.get("gallery", {})
        model_info = manifest.get("model", {})
        expected_gallery_hash = str(gallery_info.get("sha256", "")).upper()
        expected_model_hash = str(model_info.get("sha256", "")).upper()
        if not expected_gallery_hash or sha256_file(gallery_file) != expected_gallery_hash:
            raise ModelIntegrityError("embedding gallery SHA-256 mismatch")
        if not expected_model_hash:
            raise ModelIntegrityError("embedding model SHA-256 is missing")
        class_table_path = gallery_info.get("class_table_path")
        expected_class_table_hash = str(gallery_info.get("class_table_sha256", "")).upper()
        if class_table_path or expected_class_table_hash:
            if not class_table_path or not expected_class_table_hash:
                raise ModelIntegrityError("embedding class-table manifest is incomplete")
            class_table_file = manifest_file.parent / str(class_table_path)
            if sha256_file(class_table_file) != expected_class_table_hash:
                raise ModelIntegrityError("embedding class-table SHA-256 mismatch")
        with np.load(gallery_file, allow_pickle=False) as payload:
            visual_group_ids = payload["visual_group_ids"] if "visual_group_ids" in payload else None
            upgrade_counts = payload["upgrade_counts"] if "upgrade_counts" in payload else None
            upgrade_markers = payload["upgrade_markers"] if "upgrade_markers" in payload else None
            return cls(
                payload["embeddings"],
                payload["class_names"],
                payload["card_ids"],
                model_sha256=expected_model_hash,
                gallery_sha256=expected_gallery_hash,
                visual_group_ids=visual_group_ids,
                upgrade_counts=upgrade_counts,
                upgrade_markers=upgrade_markers,
            )

    def infer_upgrade_state(
        self,
        image: Any,
        box: tuple[int, int, int, int],
        hit: EmbeddingHit,
        *,
        source_color_order: str = "RGB",
        maximum_distance: float | None = None,
        minimum_margin: float | None = None,
    ) -> bool | None:
        """Resolve normal/+ independently from the retrieved card artwork."""

        import numpy as np
        from PIL import Image

        if len(hit.candidate_card_ids or (hit.card_id,)) == 1:
            return None
        if self.upgrade_markers is None:
            raise ModelIntegrityError("gallery has no upgrade-marker templates")
        if source_color_order not in {"RGB", "BGR"}:
            raise ValueError("source_color_order must be RGB or BGR")
        if maximum_distance is not None and maximum_distance <= 0.0:
            raise ValueError("maximum_distance must be positive")
        if minimum_margin is not None and minimum_margin < 0.0:
            raise ValueError("minimum_margin must be non-negative")
        array = np.asarray(image)
        x, y, width, height = box
        frame_height, frame_width = array.shape[:2]
        if width < 8 or height < 8 or x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
            raise ValueError("candidate ROI is missing, severely cropped, or outside the frame")
        crop = array[y : y + height, x : x + width, :3]
        if source_color_order == "BGR":
            crop = crop[:, :, ::-1]
        query = extract_upgrade_marker(Image.fromarray(np.ascontiguousarray(crop, dtype=np.uint8), mode="RGB"))
        scores: dict[int, float] = {}
        for upgrade in {0, 1}:
            indices = [
                index
                for index, (group, value) in enumerate(zip(self.visual_group_ids, self.upgrade_counts, strict=True))
                if group == hit.visual_group_id and value == upgrade
            ]
            if indices:
                scores[upgrade] = min(
                    float(np.mean((query - self.upgrade_markers[index]) ** 2)) for index in indices
                )
        if set(scores) != {0, 1}:
            raise ValueError("retrieved artwork group does not contain one normal/+ pair")
        best_state = min(scores, key=scores.get)
        alternate_state = 1 - best_state
        if maximum_distance is not None and scores[best_state] > maximum_distance:
            return None
        if minimum_margin is not None and scores[alternate_state] - scores[best_state] < minimum_margin:
            return None
        return best_state == 1

    def search(self, query_embedding: Any, *, top_k: int = 5) -> tuple[EmbeddingHit, ...]:
        """Return the best gallery variant for each of the top business-card IDs."""

        import numpy as np

        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape != (self.EMBEDDING_DIM,) or not np.isfinite(query).all():
            raise ValueError("query embedding must be one finite 128-dimensional vector")
        norm = float(np.linalg.norm(query))
        if norm <= 1e-8:
            raise ValueError("query embedding must be non-zero")
        scores = self.embeddings @ (query / norm)

        best_by_group: dict[str, EmbeddingHit] = {}
        for index in np.argsort(-scores, kind="stable"):
            source_index = int(index)
            card_id = self.card_ids[source_index]
            group_id = self.visual_group_ids[source_index]
            if group_id not in best_by_group:
                members = sorted(
                    {
                        (member_id, upgrade)
                        for member_group, member_id, upgrade in zip(
                            self.visual_group_ids,
                            self.card_ids,
                            self.upgrade_counts,
                            strict=True,
                        )
                        if member_group == group_id
                    },
                    key=lambda item: int(item[0]),
                )
                best_by_group[group_id] = EmbeddingHit(
                    card_id=card_id,
                    class_name=self.class_names[source_index],
                    similarity=float(scores[source_index]),
                    source_index=source_index,
                    visual_group_id=group_id,
                    candidate_card_ids=tuple(item[0] for item in members),
                    candidate_upgrade_counts=tuple(item[1] for item in members),
                )
                if len(best_by_group) == top_k:
                    break
        return tuple(best_by_group.values())


class OnnxCardEmbedder:
    """Pinned 64x64 RGB ONNX encoder for card-art retrieval."""

    INPUT_SIZE = 64
    EMBEDDING_DIM = 128

    def __init__(
        self,
        model_path: str | Path,
        *,
        expected_model_sha256: str | None = None,
        source_color_order: str = "BGR",
    ) -> None:
        self.model_path = Path(model_path)
        self.model_sha256 = sha256_file(self.model_path)
        if expected_model_sha256 and self.model_sha256 != expected_model_sha256.upper():
            raise ModelIntegrityError(f"model SHA-256 mismatch: {self.model_sha256}")
        if source_color_order not in {"BGR", "RGB"}:
            raise ValueError("source_color_order must be BGR or RGB")
        self.source_color_order = source_color_order
        self._session: Any | None = None

    def _load_session(self) -> Any:
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise RuntimeError("onnxruntime is required for card embedding") from error
            self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            if len(inputs) != 1 or inputs[0].name != "input" or list(inputs[0].shape[1:]) != [3, 64, 64]:
                raise ModelIntegrityError("unexpected embedding ONNX input contract")
            if len(outputs) != 1 or outputs[0].name != "embedding" or outputs[0].shape[-1] != self.EMBEDDING_DIM:
                raise ModelIntegrityError("unexpected embedding ONNX output contract")
        return self._session

    def preprocess(self, image: Any, box: tuple[int, int, int, int]) -> Any:
        """Convert an in-memory BGR/RGB frame ROI to one 64x64 RGB NCHW tensor."""

        try:
            import numpy as np
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("numpy and Pillow are required for card embedding preprocessing") from error

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError("frame must be HxWx3 or HxWx4")
        x, y, width, height = box
        frame_height, frame_width = array.shape[:2]
        if width < 8 or height < 8 or x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
            raise ValueError("candidate ROI is missing, severely cropped, or outside the frame")
        crop = array[y : y + height, x : x + width, :3]
        if self.source_color_order == "BGR":
            crop = crop[:, :, ::-1]
        focused = focus_card_art(Image.fromarray(np.ascontiguousarray(crop, dtype=np.uint8), mode="RGB"))
        resized = focused.resize(
            (self.INPUT_SIZE, self.INPUT_SIZE), Image.Resampling.BILINEAR
        )
        rgb = np.asarray(resized, dtype=np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[None, ...]

    def embed(self, image: Any, box: tuple[int, int, int, int]) -> Any:
        return self.embed_tensor(self.preprocess(image, box))

    def embed_tensor(self, tensor: Any) -> Any:
        """Run one already-canonicalised 64x64 RGB NCHW tensor."""

        import numpy as np

        query = np.asarray(tensor, dtype=np.float32)
        if query.shape != (1, 3, self.INPUT_SIZE, self.INPUT_SIZE) or not np.isfinite(query).all():
            raise ValueError("embedding tensor must be one finite 1x3x64x64 array")
        output = self._load_session().run(["embedding"], {"input": query})[0]
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        if vector.shape != (self.EMBEDDING_DIM,) or not np.isfinite(vector).all():
            raise ModelIntegrityError("embedding output is malformed")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            raise ModelIntegrityError("embedding output is zero")
        return vector / norm


class EmbeddingCardRecognizer:
    """Production adapter from Top-5 embedding retrieval to ``CardPrediction``."""

    NAME = "card_art_embedding_top5_v1"

    def __init__(
        self,
        embedder: OnnxCardEmbedder,
        gallery: EmbeddingGallery,
        *,
        classes_sha256: str,
        upgrade_maximum_distance: float,
        upgrade_minimum_margin: float,
    ) -> None:
        if embedder.model_sha256 != gallery.model_sha256:
            raise ModelIntegrityError("embedding model and gallery manifest disagree")
        if upgrade_maximum_distance <= 0.0 or upgrade_minimum_margin < 0.0:
            raise ValueError("invalid upgrade-state rejection thresholds")
        self.embedder = embedder
        self.gallery = gallery
        self.model_sha256 = embedder.model_sha256
        self.gallery_sha256 = gallery.gallery_sha256
        self.classes_sha256 = classes_sha256.upper()
        self.upgrade_maximum_distance = float(upgrade_maximum_distance)
        self.upgrade_minimum_margin = float(upgrade_minimum_margin)

    @classmethod
    def load(cls, model_root: str | Path, *, source_color_order: str = "BGR") -> EmbeddingCardRecognizer:
        root = Path(model_root)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("schema_version") != 1:
            raise ModelIntegrityError("unsupported production embedding manifest schema")
        model = manifest.get("model", {})
        gallery_info = manifest.get("gallery", {})
        runtime = manifest.get("runtime", {})
        class_table_path = root / str(gallery_info.get("class_table_path", ""))
        classes_sha256 = str(gallery_info.get("class_table_sha256", "")).upper()
        if not classes_sha256 or sha256_file(class_table_path) != classes_sha256:
            raise ModelIntegrityError("embedding class-table SHA-256 mismatch")
        class_table = json.loads(class_table_path.read_text(encoding="utf-8-sig"))
        if (
            not isinstance(class_table, list)
            or not all(isinstance(value, str) and value.split("_", 1)[0].isdigit() for value in class_table)
            or len(class_table) != len(set(class_table))
        ):
            raise ModelIntegrityError("embedding class table is invalid")
        gallery = EmbeddingGallery.load(
            root / str(gallery_info.get("path", "")),
            manifest_path,
        )
        if tuple(class_table) != gallery.class_names:
            raise ModelIntegrityError("embedding class table and gallery order disagree")
        declared_ids = _declared_business_card_ids(manifest.get("source"))
        actual_ids = frozenset(gallery.card_ids)
        if actual_ids != declared_ids:
            missing = sorted(declared_ids - actual_ids, key=int)
            surplus = sorted(actual_ids - declared_ids, key=int)
            raise ModelIntegrityError(
                f"embedding gallery business-card coverage disagrees: missing={missing}, surplus={surplus}"
            )
        visual_identity_count = manifest["source"].get("visual_identity_count")
        if (
            isinstance(visual_identity_count, bool)
            or not isinstance(visual_identity_count, int)
            or visual_identity_count != len(set(gallery.visual_group_ids))
        ):
            raise ModelIntegrityError("embedding visual-identity count disagrees with the gallery")
        return cls(
            OnnxCardEmbedder(
                root / str(model.get("path", "")),
                expected_model_sha256=str(model.get("sha256", "")),
                source_color_order=source_color_order,
            ),
            gallery,
            classes_sha256=classes_sha256,
            upgrade_maximum_distance=float(runtime["upgrade_maximum_distance"]),
            upgrade_minimum_margin=float(runtime["upgrade_minimum_margin"]),
        )

    def classify(
        self,
        image: Any,
        candidate: CandidateBox,
        *,
        frame_id: str,
        plan: str,
        acceptance_threshold: float | None,
        min_margin: float | None,
        resolve_upgrade_state: bool = True,
        top_k: int = 5,
    ) -> CardPrediction:
        return self.classify_embedding(
            self.embedder.embed(image, candidate.box),
            candidate,
            frame_id=frame_id,
            plan=plan,
            acceptance_threshold=acceptance_threshold,
            min_margin=min_margin,
            resolve_upgrade_state=resolve_upgrade_state,
            upgrade_image=image,
            top_k=top_k,
        )

    def classify_embedding(
        self,
        query_embedding: Any,
        candidate: CandidateBox,
        *,
        frame_id: str,
        plan: str,
        acceptance_threshold: float | None,
        min_margin: float | None,
        resolve_upgrade_state: bool = False,
        upgrade_image: Any | None = None,
        top_k: int = 5,
    ) -> CardPrediction:
        """Classify a canonical embedding while preserving the production audit contract."""
        if acceptance_threshold is not None and not 0.0 < acceptance_threshold <= 1.0:
            raise ValueError("acceptance_threshold must be in (0,1]")
        if min_margin is not None and not 0.0 <= min_margin <= 1.0:
            raise ValueError("min_margin must be in [0,1]")
        if top_k < 2:
            raise ValueError("top_k must be at least 2")
        hits = self.gallery.search(query_embedding, top_k=top_k)
        similarity = hits[0].similarity
        margin = similarity - hits[1].similarity
        top_k_card_ids = tuple(hit.candidate_card_ids or (hit.card_id,) for hit in hits)
        top_k_scores = tuple(hit.similarity for hit in hits)
        upgraded = False
        if resolve_upgrade_state:
            if upgrade_image is None:
                raise ValueError("upgrade_image is required when resolving upgrade state")
            upgraded = self.gallery.infer_upgrade_state(
                upgrade_image,
                candidate.box,
                hits[0],
                source_color_order=self.embedder.source_color_order,
                maximum_distance=self.upgrade_maximum_distance,
                minimum_margin=self.upgrade_minimum_margin,
            )
        try:
            exact = resolve_card_identity(hits, plan=plan, upgraded=upgraded)
        except ValueError as error:
            message = str(error)
            reason = "plan_type_unresolved" if "plan type" in message else "upgrade_state_unresolved"
            return self._prediction(
                candidate,
                frame_id=frame_id,
                card_id=UNKNOWN,
                predicted_card_id=UNKNOWN,
                class_name=hits[0].class_name,
                similarity=similarity,
                margin=margin,
                accepted=False,
                reason=reason,
                top_k_card_ids=top_k_card_ids,
                top_k_scores=top_k_scores,
            )

        accepted = True
        reason = "accepted" if resolve_upgrade_state else "accepted_visual_family"
        if acceptance_threshold is None or min_margin is None:
            accepted = False
            reason = "threshold_not_calibrated"
        elif similarity < acceptance_threshold:
            accepted = False
            reason = "below_acceptance_threshold"
        elif margin < min_margin:
            accepted = False
            reason = "below_margin_threshold"
        class_name = next(
            name
            for name, card_id in zip(self.gallery.class_names, self.gallery.card_ids, strict=True)
            if card_id == exact.card_id
        )
        return self._prediction(
            candidate,
            frame_id=frame_id,
            card_id=exact.card_id if accepted else UNKNOWN,
            predicted_card_id=exact.card_id,
            class_name=class_name,
            similarity=similarity,
            margin=margin,
            accepted=accepted,
            reason=reason,
            top_k_card_ids=top_k_card_ids,
            top_k_scores=top_k_scores,
        )

    def _prediction(
        self,
        candidate: CandidateBox,
        *,
        frame_id: str,
        card_id: str,
        predicted_card_id: str,
        class_name: str,
        similarity: float,
        margin: float,
        accepted: bool,
        reason: str,
        top_k_card_ids: tuple[tuple[str, ...], ...],
        top_k_scores: tuple[float, ...],
    ) -> CardPrediction:
        return CardPrediction(
            slot=candidate.slot,
            box=candidate.box,
            frame_id=frame_id,
            card_id=card_id,
            predicted_card_id=predicted_card_id,
            class_name=class_name,
            confidence=similarity,
            margin=margin,
            raw_score=similarity,
            accepted=accepted,
            reason=reason,
            model_sha256=self.model_sha256,
            classes_sha256=self.classes_sha256,
            detector_label=candidate.detector_label,
            detector_score=candidate.detector_score,
            visible=candidate.visible,
            clickable=candidate.clickable,
            source_frames=(frame_id,),
            recognizer=self.NAME,
            gallery_sha256=self.gallery_sha256,
            top_k_card_ids=top_k_card_ids,
            top_k_scores=top_k_scores,
        )
