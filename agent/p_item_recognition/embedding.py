from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from collections.abc import Sequence

try:
    from agent.card_selection.model import ModelIntegrityError, sha256_file
    from agent.card_selection.embedding import EmbeddingGallery
except ModuleNotFoundError as error:
    if error.name != "agent":
        raise
    from card_selection.model import ModelIntegrityError, sha256_file
    from card_selection.embedding import EmbeddingGallery

from .types import (
    UNKNOWN_P_ITEM_ID,
    PItemHit,
    PItemPrediction,
    ResolvedPItemIdentity,
)
from .preprocess import focus_p_item_art, extract_p_item_upgrade_marker

VALID_PLAN_TYPES = frozenset(("sense", "logic", "anomaly"))
PLAN_AMBIGUITY_IDS = frozenset(("406", "407", "408"))
PLAN_AMBIGUITY_MAPPING = {"sense": "406", "logic": "407", "anomaly": "408"}


def resolve_p_item_identity(
    hit: PItemHit,
    *,
    plan: str,
    upgraded: bool | None,
) -> ResolvedPItemIdentity:
    candidate_ids = hit.candidate_p_item_ids or (hit.p_item_id,)
    upgrade_states = hit.candidate_upgrade_states or tuple(0 for _ in candidate_ids)
    if len(candidate_ids) != len(upgrade_states):
        raise ValueError("visual-group P-item IDs and upgrade states are inconsistent")
    normalised_plan = str(plan).strip().lower()
    if len(candidate_ids) == 1:
        return ResolvedPItemIdentity(
            p_item_id=candidate_ids[0],
            visual_group_id=hit.visual_group_id,
            plan=normalised_plan,
            upgraded=None,
            plan_resolved=False,
            upgrade_resolved=False,
        )
    if set(candidate_ids) == PLAN_AMBIGUITY_IDS:
        if normalised_plan not in VALID_PLAN_TYPES:
            raise ValueError("reliable plan type is required for P-item IDs 406/407/408")
        return ResolvedPItemIdentity(
            p_item_id=PLAN_AMBIGUITY_MAPPING[normalised_plan],
            visual_group_id=hit.visual_group_id,
            plan=normalised_plan,
            upgraded=None,
            plan_resolved=True,
            upgrade_resolved=False,
        )
    if upgraded is None:
        raise ValueError("reliable upgrade state is required for a multi-ID P-item artwork group")
    wanted = int(upgraded)
    eligible = [
        p_item_id
        for p_item_id, state in zip(candidate_ids, upgrade_states, strict=True)
        if state == wanted
    ]
    if len(eligible) != 1:
        raise ValueError("upgrade state does not resolve the P-item artwork group uniquely")
    return ResolvedPItemIdentity(
        p_item_id=eligible[0],
        visual_group_id=hit.visual_group_id,
        plan=normalised_plan,
        upgraded=upgraded,
        plan_resolved=False,
        upgrade_resolved=True,
    )


class PItemEmbeddingGallery:
    EMBEDDING_DIM = 128

    def __init__(
        self,
        embeddings: Any,
        class_names: Any,
        p_item_ids: Any,
        visual_group_ids: Any,
        upgrade_states: Any,
        upgrade_markers: Any,
        plans: Any,
        *,
        model_sha256: str,
        gallery_sha256: str = "IN_MEMORY",
    ) -> None:
        import numpy as np

        self._engine = EmbeddingGallery(
            embeddings,
            class_names,
            p_item_ids,
            model_sha256=model_sha256,
            gallery_sha256=gallery_sha256,
            visual_group_ids=visual_group_ids,
            upgrade_counts=upgrade_states,
        )
        markers = np.asarray(upgrade_markers, dtype=np.float32)
        plans_array = np.asarray(plans)
        expected_count = len(self._engine.class_names)
        if markers.shape != (expected_count, 24, 24, 3) or not np.isfinite(markers).all():
            raise ModelIntegrityError("P-item upgrade-marker templates are malformed")
        if plans_array.shape != (expected_count,):
            raise ModelIntegrityError("P-item plan metadata is malformed")
        normalised_plans = tuple(str(value) for value in plans_array)
        if any(value not in VALID_PLAN_TYPES | {"free"} for value in normalised_plans):
            raise ModelIntegrityError("P-item plan metadata contains an unsupported value")
        if len(set(self._engine.card_ids)) != 420:
            raise ModelIntegrityError("P-item gallery must cover exactly 420 business IDs")
        if len(set(self._engine.visual_group_ids)) != 273:
            raise ModelIntegrityError("P-item gallery must contain exactly 273 visual identities")
        self.upgrade_markers = np.ascontiguousarray(markers)
        self.plans = normalised_plans
        self.model_sha256 = self._engine.model_sha256
        self.gallery_sha256 = self._engine.gallery_sha256

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._engine.class_names

    @property
    def p_item_ids(self) -> tuple[str, ...]:
        return self._engine.card_ids

    @property
    def visual_group_ids(self) -> tuple[str, ...]:
        return self._engine.visual_group_ids

    @property
    def upgrade_states(self) -> tuple[int, ...]:
        return self._engine.upgrade_counts

    @classmethod
    def load(
        cls,
        gallery_path: str | Path,
        manifest_path: str | Path,
    ) -> PItemEmbeddingGallery:
        import numpy as np

        gallery_file = Path(gallery_path)
        manifest_file = Path(manifest_path)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
        if manifest.get("schema_version") != 1:
            raise ModelIntegrityError("unsupported P-item embedding manifest schema")
        gallery_info = manifest.get("gallery", {})
        model_info = manifest.get("model", {})
        expected_gallery_hash = str(gallery_info.get("sha256", "")).upper()
        expected_model_hash = str(model_info.get("sha256", "")).upper()
        if not expected_gallery_hash or sha256_file(gallery_file) != expected_gallery_hash:
            raise ModelIntegrityError("P-item embedding gallery SHA-256 mismatch")
        if not expected_model_hash:
            raise ModelIntegrityError("P-item embedding model SHA-256 is missing")
        class_table_file = manifest_file.parent / str(gallery_info.get("class_table_path", ""))
        expected_class_hash = str(gallery_info.get("class_table_sha256", "")).upper()
        if not expected_class_hash or sha256_file(class_table_file) != expected_class_hash:
            raise ModelIntegrityError("P-item embedding class-table SHA-256 mismatch")
        class_table = json.loads(class_table_file.read_text(encoding="utf-8-sig"))
        with np.load(gallery_file, allow_pickle=False) as payload:
            required = {
                "embeddings",
                "class_names",
                "p_item_ids",
                "visual_group_ids",
                "upgrade_states",
                "upgrade_markers",
                "plans",
            }
            if not required.issubset(payload.files):
                raise ModelIntegrityError("P-item gallery payload is incomplete")
            payload_names = tuple(str(value) for value in payload["class_names"])
            if class_table != list(payload_names):
                raise ModelIntegrityError("P-item gallery class table differs from its payload")
            gallery = cls(
                payload["embeddings"],
                payload_names,
                payload["p_item_ids"],
                payload["visual_group_ids"],
                payload["upgrade_states"],
                payload["upgrade_markers"],
                payload["plans"],
                model_sha256=expected_model_hash,
                gallery_sha256=expected_gallery_hash,
            )
        manifest_ids = {str(value) for value in manifest.get("source", {}).get("business_ids", [])}
        if manifest_ids and manifest_ids != set(gallery.p_item_ids):
            raise ModelIntegrityError("P-item manifest business-ID scope differs from the gallery")
        return gallery

    def search(self, query_embedding: Any, *, top_k: int = 5) -> tuple[PItemHit, ...]:
        hits = self._engine.search(query_embedding, top_k=top_k)
        return tuple(
            PItemHit(
                p_item_id=hit.card_id,
                class_name=hit.class_name,
                similarity=hit.similarity,
                source_index=hit.source_index,
                visual_group_id=hit.visual_group_id,
                candidate_p_item_ids=hit.candidate_card_ids,
                candidate_upgrade_states=hit.candidate_upgrade_counts,
            )
            for hit in hits
        )

    def infer_upgrade_state(
        self,
        image: Any,
        box: tuple[int, int, int, int],
        hit: PItemHit,
        *,
        source_color_order: str,
        maximum_distance: float,
        minimum_margin: float,
    ) -> bool | None:
        import numpy as np
        from PIL import Image

        candidate_ids = hit.candidate_p_item_ids or (hit.p_item_id,)
        if len(candidate_ids) == 1 or set(candidate_ids) == PLAN_AMBIGUITY_IDS:
            return None
        if source_color_order not in {"RGB", "BGR"}:
            raise ValueError("source_color_order must be RGB or BGR")
        if maximum_distance <= 0.0 or minimum_margin < 0.0:
            raise ValueError("invalid P-item upgrade-state thresholds")
        array = np.asarray(image)
        x, y, width, height = box
        frame_height, frame_width = array.shape[:2]
        if (
            array.ndim != 3
            or array.shape[2] not in {3, 4}
            or width < 8
            or height < 8
            or x < 0
            or y < 0
            or x + width > frame_width
            or y + height > frame_height
        ):
            raise ValueError("P-item candidate ROI is missing, cropped, or outside the frame")
        crop = array[y : y + height, x : x + width, :3]
        if source_color_order == "BGR":
            crop = crop[:, :, ::-1]
        query = extract_p_item_upgrade_marker(
            Image.fromarray(np.ascontiguousarray(crop, dtype=np.uint8), mode="RGB")
        )
        scores: dict[int, float] = {}
        for state in (0, 1):
            indices = [
                index
                for index, (group, value) in enumerate(
                    zip(self.visual_group_ids, self.upgrade_states, strict=True)
                )
                if group == hit.visual_group_id and value == state
            ]
            if indices:
                scores[state] = min(
                    float(np.mean((query - self.upgrade_markers[index]) ** 2))
                    for index in indices
                )
        if set(scores) != {0, 1}:
            raise ValueError("retrieved P-item artwork group has no normal/+ pair")
        best_state = min(scores, key=scores.get)
        if scores[best_state] > maximum_distance:
            return None
        if scores[1 - best_state] - scores[best_state] < minimum_margin:
            return None
        return best_state == 1


class OnnxPItemEmbedder:
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
            raise ModelIntegrityError("P-item embedding model SHA-256 mismatch")
        if source_color_order not in {"RGB", "BGR"}:
            raise ValueError("source_color_order must be RGB or BGR")
        self.source_color_order = source_color_order
        self._session: Any | None = None

    def _load_session(self) -> Any:
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise RuntimeError("onnxruntime is required for P-item embedding") from error
            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            if (
                len(inputs) != 1
                or inputs[0].name != "input"
                or list(inputs[0].shape[1:]) != [3, 64, 64]
            ):
                raise ModelIntegrityError("unexpected P-item embedding ONNX input contract")
            if (
                len(outputs) != 1
                or outputs[0].name != "embedding"
                or outputs[0].shape[-1] != self.EMBEDDING_DIM
            ):
                raise ModelIntegrityError("unexpected P-item embedding ONNX output contract")
        return self._session

    def preprocess(self, image: Any, box: tuple[int, int, int, int]) -> Any:
        import numpy as np
        from PIL import Image

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError("frame must be HxWx3 or HxWx4")
        x, y, width, height = box
        frame_height, frame_width = array.shape[:2]
        if (
            width < 8
            or height < 8
            or x < 0
            or y < 0
            or x + width > frame_width
            or y + height > frame_height
        ):
            raise ValueError("P-item candidate ROI is missing, cropped, or outside the frame")
        crop = array[y : y + height, x : x + width, :3]
        if self.source_color_order == "BGR":
            crop = crop[:, :, ::-1]
        focused = focus_p_item_art(
            Image.fromarray(np.ascontiguousarray(crop, dtype=np.uint8), mode="RGB")
        ).resize((self.INPUT_SIZE, self.INPUT_SIZE), Image.Resampling.BILINEAR)
        rgb = np.asarray(focused, dtype=np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[None, ...]

    def embed(self, image: Any, box: tuple[int, int, int, int]) -> Any:
        return self.embed_many(((image, box),))[0]

    def embed_many(
        self,
        observations: Sequence[tuple[Any, tuple[int, int, int, int]]],
    ) -> tuple[Any, ...]:
        """Embed one stable page generation in a single ONNX batch."""

        import numpy as np

        if not observations:
            return ()
        batch = np.concatenate(
            [self.preprocess(image, box) for image, box in observations],
            axis=0,
        )
        output = self._load_session().run(
            ["embedding"], {"input": batch}
        )[0]
        vectors = np.asarray(output, dtype=np.float32)
        if (
            vectors.shape != (len(observations), self.EMBEDDING_DIM)
            or not np.isfinite(vectors).all()
        ):
            raise ModelIntegrityError("P-item embedding output is malformed")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 1e-8):
            raise ModelIntegrityError("P-item embedding output is zero")
        normalised = vectors / norms[:, None]
        return tuple(normalised[index] for index in range(len(observations)))


class PItemEmbeddingRecognizer:
    NAME = "p_item_art_embedding_top5_v1"

    def __init__(
        self,
        embedder: OnnxPItemEmbedder,
        gallery: PItemEmbeddingGallery,
        *,
        classes_sha256: str,
        upgrade_maximum_distance: float | None,
        upgrade_minimum_margin: float | None,
    ) -> None:
        if embedder.model_sha256 != gallery.model_sha256:
            raise ModelIntegrityError("P-item model and gallery manifest disagree")
        if (upgrade_maximum_distance is None) != (upgrade_minimum_margin is None):
            raise ValueError("P-item upgrade thresholds must be supplied together")
        self.embedder = embedder
        self.gallery = gallery
        self.classes_sha256 = classes_sha256.upper()
        self.model_sha256 = embedder.model_sha256
        self.gallery_sha256 = gallery.gallery_sha256
        self.upgrade_maximum_distance = upgrade_maximum_distance
        self.upgrade_minimum_margin = upgrade_minimum_margin

    @classmethod
    def load(
        cls,
        model_root: str | Path,
        *,
        source_color_order: str = "BGR",
    ) -> PItemEmbeddingRecognizer:
        root = Path(model_root)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        model = manifest.get("model", {})
        gallery_info = manifest.get("gallery", {})
        runtime = manifest.get("runtime", {})
        class_table_path = root / str(gallery_info.get("class_table_path", ""))
        classes_sha256 = str(gallery_info.get("class_table_sha256", "")).upper()
        if not classes_sha256 or sha256_file(class_table_path) != classes_sha256:
            raise ModelIntegrityError("P-item embedding class-table SHA-256 mismatch")
        return cls(
            OnnxPItemEmbedder(
                root / str(model.get("path", "")),
                expected_model_sha256=str(model.get("sha256", "")),
                source_color_order=source_color_order,
            ),
            PItemEmbeddingGallery.load(
                root / str(gallery_info.get("path", "")), manifest_path
            ),
            classes_sha256=classes_sha256,
            upgrade_maximum_distance=(
                float(runtime["upgrade_maximum_distance"])
                if "upgrade_maximum_distance" in runtime
                else None
            ),
            upgrade_minimum_margin=(
                float(runtime["upgrade_minimum_margin"])
                if "upgrade_minimum_margin" in runtime
                else None
            ),
        )

    def classify(
        self,
        image: Any,
        box: tuple[int, int, int, int],
        *,
        frame_id: str,
        plan: str,
        acceptance_threshold: float | None,
        minimum_margin: float | None,
    ) -> PItemPrediction:
        if acceptance_threshold is not None and not 0.0 < acceptance_threshold <= 1.0:
            raise ValueError("acceptance_threshold must be in (0,1]")
        if minimum_margin is not None and not 0.0 <= minimum_margin <= 1.0:
            raise ValueError("minimum_margin must be in [0,1]")
        hits = self.shortlist(image, box, limit=5)
        if len(hits) < 2:
            raise ModelIntegrityError("P-item gallery returned fewer than two visual identities")
        similarity = hits[0].similarity
        margin = similarity - hits[1].similarity
        top_k_ids = tuple(hit.candidate_p_item_ids or (hit.p_item_id,) for hit in hits)
        top_k_scores = tuple(hit.similarity for hit in hits)
        upgraded: bool | None = None
        candidate_ids = hits[0].candidate_p_item_ids or (hits[0].p_item_id,)
        if len(candidate_ids) > 1 and set(candidate_ids) != PLAN_AMBIGUITY_IDS:
            if self.upgrade_maximum_distance is None or self.upgrade_minimum_margin is None:
                return self._prediction(
                    box,
                    frame_id=frame_id,
                    hits=hits,
                    p_item_id=UNKNOWN_P_ITEM_ID,
                    predicted_p_item_id=UNKNOWN_P_ITEM_ID,
                    similarity=similarity,
                    margin=margin,
                    accepted=False,
                    reason="upgrade_threshold_not_calibrated",
                    upgraded=None,
                    plan=plan,
                    top_k_ids=top_k_ids,
                    top_k_scores=top_k_scores,
                )
            upgraded = self.gallery.infer_upgrade_state(
                image,
                box,
                hits[0],
                source_color_order=self.embedder.source_color_order,
                maximum_distance=self.upgrade_maximum_distance,
                minimum_margin=self.upgrade_minimum_margin,
            )
        try:
            exact = resolve_p_item_identity(hits[0], plan=plan, upgraded=upgraded)
        except ValueError as error:
            reason = "plan_type_unresolved" if "plan type" in str(error) else "upgrade_state_unresolved"
            return self._prediction(
                box,
                frame_id=frame_id,
                hits=hits,
                p_item_id=UNKNOWN_P_ITEM_ID,
                predicted_p_item_id=UNKNOWN_P_ITEM_ID,
                similarity=similarity,
                margin=margin,
                accepted=False,
                reason=reason,
                upgraded=upgraded,
                plan=plan,
                top_k_ids=top_k_ids,
                top_k_scores=top_k_scores,
            )
        accepted = True
        reason = "accepted"
        if acceptance_threshold is None or minimum_margin is None:
            accepted = False
            reason = "threshold_not_calibrated"
        elif similarity < acceptance_threshold:
            accepted = False
            reason = "below_acceptance_threshold"
        elif margin < minimum_margin:
            accepted = False
            reason = "below_margin_threshold"
        return self._prediction(
            box,
            frame_id=frame_id,
            hits=hits,
            p_item_id=exact.p_item_id if accepted else UNKNOWN_P_ITEM_ID,
            predicted_p_item_id=exact.p_item_id,
            similarity=similarity,
            margin=margin,
            accepted=accepted,
            reason=reason,
            upgraded=exact.upgraded,
            plan=exact.plan,
            top_k_ids=top_k_ids,
            top_k_scores=top_k_scores,
        )

    def shortlist(
        self,
        image: Any,
        box: tuple[int, int, int, int],
        *,
        limit: int,
    ) -> tuple[PItemHit, ...]:
        """Return visual-identity candidates without making an ID decision.

        The shortlist is a retrieval primitive only.  Callers must still use
        the fixed rendered gallery (or an explicitly approved equivalent) to
        decide the final business ID.
        """

        if limit < 1:
            raise ValueError("shortlist limit must be positive")
        return self.gallery.search(self.embedder.embed(image, box), top_k=limit)

    def shortlists(
        self,
        observations: Sequence[tuple[Any, tuple[int, int, int, int]]],
        *,
        limit: int,
    ) -> tuple[tuple[PItemHit, ...], ...]:
        """Retrieve candidates for a stable page generation in one ONNX batch."""

        if limit < 1:
            raise ValueError("shortlist limit must be positive")
        return tuple(
            self.gallery.search(vector, top_k=limit)
            for vector in self.embedder.embed_many(observations)
        )

    def _prediction(
        self,
        box: tuple[int, int, int, int],
        *,
        frame_id: str,
        hits: tuple[PItemHit, ...],
        p_item_id: str,
        predicted_p_item_id: str,
        similarity: float,
        margin: float,
        accepted: bool,
        reason: str,
        upgraded: bool | None,
        plan: str,
        top_k_ids: tuple[tuple[str, ...], ...],
        top_k_scores: tuple[float, ...],
    ) -> PItemPrediction:
        return PItemPrediction(
            box=box,
            frame_id=frame_id,
            p_item_id=p_item_id,
            predicted_p_item_id=predicted_p_item_id,
            visual_group_id=hits[0].visual_group_id,
            confidence=similarity,
            margin=margin,
            accepted=accepted,
            reason=reason,
            upgraded=upgraded,
            plan=str(plan).strip().lower(),
            model_sha256=self.model_sha256,
            gallery_sha256=self.gallery_sha256,
            classes_sha256=self.classes_sha256,
            recognizer=self.NAME,
            top_k_p_item_ids=top_k_ids,
            top_k_scores=top_k_scores,
        )
