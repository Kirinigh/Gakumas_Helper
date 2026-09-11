"""Compact clean-card identity gallery used by the arena reader.

The historical asset directory retains ``arena_badge_reference`` for package
compatibility, but this runtime module has no badge-presence or badge-residual
API. Customization presence is owned exclusively by ``customization_badge``.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any
from pathlib import Path
from collections.abc import Sequence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


class BadgeReferenceError(RuntimeError):
    """The compact clean-card identity asset is invalid or inconclusive."""


REFERENCE_RULE_VERSION = "task410-r10-card-identity-only-v1"


class BadgeReferenceGallery:
    """Resolve clean-card visual families without inspecting badge pixels."""

    def __init__(
        self,
        *,
        business_ids: Any,
        visual_group_ids: Any,
        reference_names: Any,
        top_coarse: Any,
        maximum_coarse_error: float,
        minimum_group_margin: float,
        maximum_zero_coarse_error: float | None = None,
        minimum_high_error_group_margin: float = 3.0,
    ) -> None:
        import numpy as np

        self.business_ids = np.asarray(business_ids, dtype=np.int16)
        self.visual_group_ids = tuple(str(value) for value in visual_group_ids)
        self.reference_names = tuple(str(value) for value in reference_names)
        self.top_coarse = np.asarray(top_coarse, dtype=np.uint8)
        count = len(self.business_ids)
        if (
            count < 1
            or len(self.visual_group_ids) != count
            or len(self.reference_names) != count
            or self.top_coarse.shape != (count, 8, 16, 3)
        ):
            raise BadgeReferenceError("card reference arrays are inconsistent")
        zero_ceiling = (
            maximum_coarse_error
            if maximum_zero_coarse_error is None
            else maximum_zero_coarse_error
        )
        if (
            maximum_coarse_error <= 0
            or zero_ceiling < maximum_coarse_error
            or minimum_group_margin < 0
            or minimum_high_error_group_margin < 0
        ):
            raise BadgeReferenceError("card reference thresholds are invalid")
        self.maximum_coarse_error = float(maximum_coarse_error)
        self.maximum_zero_coarse_error = float(zero_ceiling)
        self.minimum_group_margin = float(minimum_group_margin)
        self.minimum_high_error_group_margin = float(
            minimum_high_error_group_margin
        )
        groups: dict[str, list[int]] = {}
        for index, group in enumerate(self.visual_group_ids):
            groups.setdefault(group, []).append(index)
        self._group_indices = {
            group: np.asarray(indices, dtype=np.int32)
            for group, indices in groups.items()
        }
        self._eligible_scopes: dict[frozenset[int], tuple[Any, Any, dict[str, Any]]] = {}

    def _eligible_scope(self, eligible_card_ids: frozenset[int]):
        import numpy as np

        key = frozenset(eligible_card_ids)
        if not key:
            raise BadgeReferenceError("eligible card IDs contain no clean-card references")
        cached = self._eligible_scopes.get(key)
        if cached is not None:
            return cached
        indices = np.asarray([
            index for index, card_id in enumerate(self.business_ids) if int(card_id) in key
        ], dtype=np.int32)
        if not len(indices):
            raise BadgeReferenceError("eligible card IDs contain no clean-card references")
        groups: dict[str, list[int]] = {}
        for local_index, original_index in enumerate(indices):
            groups.setdefault(self.visual_group_ids[original_index], []).append(local_index)
        cached = (
            indices,
            self.top_coarse[indices].astype(np.int16),
            {group: np.asarray(members, dtype=np.int32) for group, members in groups.items()},
        )
        self._eligible_scopes[key] = cached
        return cached

    @classmethod
    def load(cls, root: str | Path) -> "BadgeReferenceGallery":
        import numpy as np

        model_root = Path(root)
        manifest_path = model_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("schema_version") != 1:
            raise BadgeReferenceError("unsupported card reference manifest schema")
        gallery_info = manifest.get("gallery", {})
        gallery_path = model_root / str(gallery_info.get("path", ""))
        expected_hash = str(gallery_info.get("sha256", "")).upper()
        if not expected_hash or _sha256_file(gallery_path) != expected_hash:
            raise BadgeReferenceError("card reference gallery SHA-256 mismatch")
        runtime = manifest.get("runtime", {})
        if runtime.get("reference_rule_version") != REFERENCE_RULE_VERSION:
            raise BadgeReferenceError("card reference rule version mismatch")
        with np.load(gallery_path, allow_pickle=False) as payload:
            return cls(
                business_ids=payload["business_ids"],
                visual_group_ids=payload["visual_group_ids"],
                reference_names=payload["reference_names"],
                top_coarse=payload["top_coarse"],
                maximum_coarse_error=float(runtime["maximum_coarse_error"]),
                minimum_group_margin=float(runtime["minimum_group_margin"]),
                maximum_zero_coarse_error=float(
                    runtime["maximum_zero_coarse_error"]
                ),
                minimum_high_error_group_margin=float(
                    runtime["minimum_high_error_group_margin"]
                ),
            )

    def _identity_result_from_errors(
        self,
        errors: Any,
        eligible_card_ids: frozenset[int] | None = None,
    ) -> dict[str, Any]:
        """Rank errors in the corresponding cached scope's row order."""
        import numpy as np

        if eligible_card_ids is None:
            reference_indices, selected_group_indices = None, self._group_indices
        else:
            reference_indices, _, selected_group_indices = self._eligible_scope(eligible_card_ids)
        ranked_groups = sorted(
            (
                (float(np.min(errors[indices])), group)
                for group, indices in selected_group_indices.items()
            ),
            key=lambda item: (item[0], item[1]),
        )
        top_error, top_group = ranked_groups[0]
        group_margin = (
            float("inf")
            if len(ranked_groups) < 2
            else ranked_groups[1][0] - top_error
        )
        if top_error > self.maximum_zero_coarse_error:
            return {
                "status": "AMBIGUOUS_REFERENCE_IDENTITY",
                "top_group": top_group,
                "top_error": round(top_error, 6),
                "group_margin": round(group_margin, 6),
            }
        low_confidence_identity = bool(
            top_error > self.maximum_coarse_error
            and (
                len(ranked_groups) < 2
                or group_margin < self.minimum_high_error_group_margin
            )
        )
        candidate_groups = (
            ranked_groups[:1]
            if low_confidence_identity or group_margin >= self.minimum_group_margin
            else tuple(
                item
                for item in ranked_groups
                if item[0] - top_error <= self.minimum_group_margin
            )
        )
        measured = []
        for candidate_error, candidate_group in candidate_groups:
            group_indices = selected_group_indices[candidate_group]
            reference_index = int(
                group_indices[int(np.argmin(errors[group_indices]))]
            )
            if reference_indices is not None:
                reference_index = int(reference_indices[reference_index])
            result = {
                "status": "MEASURED",
                "reference_name": self.reference_names[reference_index],
                "reference_business_id": int(self.business_ids[reference_index]),
                "reference_visual_group": candidate_group,
                "identity_top_error": round(candidate_error, 6),
                "identity_group_margin": round(group_margin, 6),
                "identity_low_confidence": low_confidence_identity,
                "identity_resolution": "coarse_identity_only",
            }
            measured.append(result)
        if len(measured) == 1:
            return measured[0]
        return {
            "status": "MEASURED_CANDIDATES",
            "top_group": top_group,
            "top_error": round(top_error, 6),
            "group_margin": round(group_margin, 6),
            "candidates": tuple(measured),
            "identity_resolution": "coarse_identity_candidates_only",
        }

    def content_signature(
        self,
        image: Any,
        box: tuple[int, int, int, int],
        *,
        eligible_card_ids: frozenset[int] | None = None,
    ) -> tuple[str, Any]:
        """Return a cheap temporal signature for one visible card."""

        visual_group, content_query, _ = self.content_signature_with_identity(
            image,
            box,
            eligible_card_ids=eligible_card_ids,
        )
        return visual_group, content_query

    def content_signature_with_identity(
        self,
        image: Any,
        box: tuple[int, int, int, int],
        *,
        eligible_card_ids: frozenset[int] | None = None,
    ) -> tuple[str, Any, dict[str, Any]]:
        """Return temporal pixels and coarse identity from one shared pass."""

        import cv2
        import numpy as np

        array = np.asarray(image)
        x, y, width, height = box
        crop = array[y : y + height, x : x + width, :3]
        if crop.shape[:2] != (height, width) or width < 8 or height < 8:
            raise BadgeReferenceError("card content signature ROI is invalid")
        live = cv2.resize(
            np.ascontiguousarray(crop),
            (96, 96),
            interpolation=cv2.INTER_AREA,
        )
        identity_query = cv2.resize(
            live[:48],
            (16, 8),
            interpolation=cv2.INTER_AREA,
        ).astype(np.int16)
        if eligible_card_ids is None:
            reference_indices, references = None, self.top_coarse.astype(np.int16)
        else:
            reference_indices, references, _ = self._eligible_scope(eligible_card_ids)
        errors = np.mean(
            np.abs(references - identity_query),
            axis=(1, 2, 3),
        )
        reference_index = int(np.argmin(errors))
        if reference_indices is not None:
            reference_index = int(reference_indices[reference_index])
        content_query = cv2.resize(
            live,
            (16, 16),
            interpolation=cv2.INTER_AREA,
        ).astype(np.uint8)
        return (
            self.visual_group_ids[reference_index],
            content_query,
            self._identity_result_from_errors(errors, eligible_card_ids),
        )

    def measure_identity(
        self,
        image: Any,
        box: tuple[int, int, int, int],
        *,
        eligible_card_ids: frozenset[int] | None = None,
    ) -> dict[str, Any]:
        """Measure card identity only; badge pixels are outside the query."""

        import cv2
        import numpy as np

        array = np.asarray(image)
        x, y, width, height = box
        crop = array[y : y + height, x : x + width, :3]
        if crop.shape[:2] != (height, width) or width < 8 or height < 8:
            return {"status": "INVALID_ROI"}
        live = cv2.resize(
            np.ascontiguousarray(crop),
            (96, 96),
            interpolation=cv2.INTER_AREA,
        )
        query = cv2.resize(
            live[:48],
            (16, 8),
            interpolation=cv2.INTER_AREA,
        ).astype(np.int16)
        references = (self.top_coarse.astype(np.int16) if eligible_card_ids is None
                      else self._eligible_scope(eligible_card_ids)[1])
        errors = np.mean(
            np.abs(references - query),
            axis=(1, 2, 3),
        )
        return self._identity_result_from_errors(errors, eligible_card_ids)


def stable_reference_business_candidates(
    references: Sequence[dict[str, Any]],
) -> tuple[int, ...]:
    """Return every high-confidence business ID shared by three frames."""

    if len(references) != 3:
        raise BadgeReferenceError(
            f"card identity requires three reference observations, found {len(references)}"
        )
    frame_candidates: list[set[int]] = []
    for reference in references:
        status = reference.get("status")
        raw_candidates: Sequence[dict[str, Any]]
        if status == "MEASURED":
            raw_candidates = (reference,)
        elif status == "MEASURED_CANDIDATES" and isinstance(
            reference.get("candidates"), Sequence
        ):
            raw_candidates = tuple(
                candidate
                for candidate in reference["candidates"]
                if isinstance(candidate, dict)
            )
        else:
            raw_candidates = ()
        candidate_ids = {
            int(candidate["reference_business_id"])
            for candidate in raw_candidates
            if candidate.get("identity_low_confidence") is not True
            and isinstance(candidate.get("reference_business_id"), int)
            and not isinstance(candidate.get("reference_business_id"), bool)
            and int(candidate["reference_business_id"]) > 0
        }
        if not candidate_ids:
            raise BadgeReferenceError(
                "fixed reference did not yield a high-confidence card identity"
            )
        frame_candidates.append(candidate_ids)
    stable_ids = set.intersection(*frame_candidates)
    if not stable_ids:
        raise BadgeReferenceError(
            "fixed reference card identity had no stable candidate across frames: "
            f"{tuple(sorted(values) for values in frame_candidates)!r}"
        )
    return tuple(sorted(stable_ids))


def stable_reference_business_identity(references: Sequence[dict[str, Any]]) -> int:
    """Return the one business ID shared by three high-confidence frames."""

    stable_ids = stable_reference_business_candidates(references)
    if len(stable_ids) != 1:
        raise BadgeReferenceError(
            "fixed reference card identity was not unique across stable frames: "
            f"{stable_ids!r}"
        )
    return stable_ids[0]
