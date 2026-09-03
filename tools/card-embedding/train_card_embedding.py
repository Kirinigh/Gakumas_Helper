from __future__ import annotations

import sys
import json
import time
import random
import hashlib
import argparse
from typing import Any
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.card_selection.embedding import CARD_ART_EXCLUSION_BOXES, focus_card_art, extract_upgrade_marker

EMBEDDING_DIM = 128
INPUT_SIZE = 64
SCHEMA_VERSION = 1
KNOWN_VISUAL_AMBIGUITY_GROUPS = (
    frozenset(("789", "790", "791", "792", "793", "794")),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def business_card_id(class_name: str) -> str:
    card_id = class_name.split("_", 1)[0]
    if not card_id.isdigit() or int(card_id) < 1:
        raise ValueError(f"invalid skill-card class name: {class_name}")
    return card_id


def natural_class_key(path: Path) -> tuple[int, int, str]:
    stem = path.stem
    prefix, separator, suffix = stem.partition("_")
    return int(prefix), int(suffix) if separator and suffix.isdigit() else -1, stem


def _positive_business_ids(values: object, *, label: str) -> frozenset[int]:
    if (
        not isinstance(values, list)
        or not values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values
        )
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{label} must contain unique positive integer business IDs")
    return frozenset(values)


def _business_ids_from_ranges(values: object, *, label: str) -> frozenset[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must contain at least one business ID range")
    business_ids: set[int] = set()
    for item in values:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in item)
            or item[0] < 1
            or item[1] < item[0]
        ):
            raise ValueError(f"{label} contains an invalid business ID range")
        current = set(range(item[0], item[1] + 1))
        if business_ids.intersection(current):
            raise ValueError(f"{label} contains overlapping business ID ranges")
        business_ids.update(current)
    return frozenset(business_ids)


def manifest_business_card_ids(manifest: dict[str, Any]) -> frozenset[int]:
    scope = manifest.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("authoritative dataset manifest has no valid business ID scope")

    declarations: list[tuple[str, frozenset[int]]] = []
    for key in ("business_card_ids", "business_ids"):
        if key in scope:
            declarations.append(
                (key, _positive_business_ids(scope[key], label=f"dataset scope {key}"))
            )
    for key in ("business_card_id_ranges", "business_id_ranges"):
        if key in scope:
            declarations.append(
                (key, _business_ids_from_ranges(scope[key], label=f"dataset scope {key}"))
            )
    for minimum_key, maximum_key in (
        ("business_card_id_min", "business_card_id_max"),
        ("business_id_min", "business_id_max"),
    ):
        minimum = scope.get(minimum_key)
        maximum = scope.get(maximum_key)
        if minimum is None and maximum is None:
            continue
        if (
            isinstance(minimum, bool)
            or isinstance(maximum, bool)
            or not isinstance(minimum, int)
            or not isinstance(maximum, int)
            or minimum < 1
            or maximum < minimum
        ):
            raise ValueError(
                f"dataset scope {minimum_key}/{maximum_key} must be one valid inclusive range"
            )
        declarations.append(
            (
                f"{minimum_key}/{maximum_key}",
                frozenset(range(minimum, maximum + 1)),
            )
        )
    if not declarations:
        raise ValueError("authoritative dataset manifest has no exact business ID set")

    first_label, expected = declarations[0]
    for label, declared in declarations[1:]:
        if declared != expected:
            raise ValueError(
                f"dataset business ID declarations disagree: {first_label} versus {label}"
            )
    for source, key in (
        (scope, "business_card_id_count"),
        (manifest.get("validation", {}), "expected_business_id_count"),
    ):
        if not isinstance(source, dict) or key not in source:
            continue
        count = source[key]
        if isinstance(count, bool) or not isinstance(count, int) or count != len(expected):
            raise ValueError(f"dataset {key} disagrees with the exact business ID set")
    return expected


def business_id_manifest_fields(values: list[int]) -> dict[str, Any]:
    business_ids = _positive_business_ids(values, label="trained model business_card_ids")
    ordered = sorted(business_ids)
    fields: dict[str, Any] = {
        "business_card_ids": ordered,
        "business_card_id_count": len(ordered),
    }
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        fields.update(
            {
                "business_card_id_min": ordered[0],
                "business_card_id_max": ordered[-1],
            }
        )
    return fields


def discover_icons(
    root: Path,
    expected_business_ids: set[int] | frozenset[int] | None = None,
) -> list[Path]:
    icons = sorted(root.glob("*.webp"), key=natural_class_key)
    if not icons:
        raise FileNotFoundError(f"no WebP icons found under {root}")
    names = [path.stem for path in icons]
    if len(set(names)) != len(names):
        raise ValueError("duplicate class names in icon source")
    ids = {int(business_card_id(name)) for name in names}
    if expected_business_ids is not None and ids != set(expected_business_ids):
        expected = set(expected_business_ids)
        raise ValueError(
            "pinned icon source business ID set differs; "
            f"missing={sorted(expected - ids)}, surplus={sorted(ids - expected)}"
        )
    return icons


def verify_dataset_manifest(manifest_path: Path, icons_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("authoritative dataset manifest must be a JSON object")
    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("authoritative dataset validation is missing; training is forbidden")
    if validation.get("status") != "CLEAR":
        raise ValueError(
            "authoritative dataset validation is not CLEAR; training is forbidden: "
            f"{validation.get('status', 'MISSING')}"
        )
    expected_ids = manifest_business_card_ids(manifest)
    icons = manifest.get("icons")
    if not isinstance(icons, dict) or not isinstance(icons.get("images"), list):
        raise ValueError("dataset manifest icon inventory is missing")
    records = icons["images"]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("dataset manifest icon records must be objects")
    raw_record_ids = [record.get("business_id") for record in records]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in raw_record_ids
    ):
        raise ValueError("dataset manifest icon records contain an invalid business ID")
    record_ids = set(raw_record_ids)
    if record_ids != expected_ids:
        raise ValueError(
            "dataset manifest icon records do not cover the exact declared business ID set; "
            f"missing={sorted(expected_ids - record_ids)}, "
            f"surplus={sorted(record_ids - expected_ids)}"
        )
    declared_count = icons.get("count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int) or len(
        records
    ) != declared_count:
        raise ValueError("dataset manifest icon record count is inconsistent")
    expected = {str(record["path"]): str(record["sha256"]).upper() for record in records}
    if len(expected) != len(records):
        raise ValueError("dataset manifest contains duplicate icon paths")
    actual = {path.name: sha256_file(path) for path in icons_root.glob("*.webp")}
    if actual != expected:
        raise ValueError("icon files or SHA-256 values differ from the authoritative dataset manifest")
    return manifest


@dataclass(frozen=True)
class TrainingSample:
    class_name: str
    business_card_id: str
    upgrade_count: int
    art_path: Path
    visual_group_id: str


def discover_art_samples(manifest: dict[str, Any], art_root: Path) -> list[TrainingSample]:
    expected_business_ids = manifest_business_card_ids(manifest)
    crosswalk_path = Path(manifest["mapping"]["crosswalk_path"])
    if not crosswalk_path.is_absolute():
        crosswalk_path = art_root.parent / crosswalk_path
    if sha256_file(crosswalk_path) != str(manifest["mapping"]["crosswalk_sha256"]).upper():
        raise ValueError("crosswalk SHA-256 differs from the authoritative dataset manifest")
    crosswalk = json.loads(crosswalk_path.read_text(encoding="utf-8"))
    if not isinstance(crosswalk, list) or not crosswalk:
        raise ValueError("authoritative crosswalk must be a non-empty list")
    crosswalk_by_id: dict[str, dict[str, Any]] = {}
    for row in crosswalk:
        if not isinstance(row, dict):
            raise ValueError("authoritative crosswalk rows must be objects")
        value = row.get("business_id")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("authoritative crosswalk contains an invalid business ID")
        card_id = str(value)
        if card_id in crosswalk_by_id:
            raise ValueError("authoritative crosswalk contains duplicate business IDs")
        internal_id = row.get("internal_id")
        variants = row.get("asset_variants")
        if (
            not isinstance(internal_id, str)
            or not internal_id
            or not isinstance(variants, list)
            or not variants
            or not all(isinstance(item, str) and item for item in variants)
            or len(variants) != len(set(variants))
        ):
            raise ValueError("authoritative crosswalk identity mapping is invalid")
        crosswalk_by_id[card_id] = row
    crosswalk_ids = {int(value) for value in crosswalk_by_id}
    if crosswalk_ids != expected_business_ids:
        raise ValueError(
            "authoritative crosswalk does not cover the exact declared business ID set; "
            f"missing={sorted(expected_business_ids - crosswalk_ids)}, "
            f"surplus={sorted(crosswalk_ids - expected_business_ids)}"
        )

    art = manifest.get("art")
    if not isinstance(art, dict) or not isinstance(art.get("images"), list):
        raise ValueError("authoritative art inventory is missing")
    art_records = art["images"]
    expected_art = {str(record["path"]): str(record["sha256"]).upper() for record in art_records}
    if len(expected_art) != len(art_records):
        raise ValueError("authoritative art inventory contains duplicate paths")
    actual_art = {path.name: sha256_file(path) for path in art_root.glob("*.webp")}
    if not expected_art or actual_art != expected_art:
        raise ValueError("authoritative art files or SHA-256 values differ from the dataset manifest")
    april_fools_ids = {"789", "790", "791", "792", "793", "794"}
    icons = manifest.get("icons")
    if not isinstance(icons, dict) or not isinstance(icons.get("images"), list):
        raise ValueError("dataset manifest icon inventory is missing")
    samples: list[TrainingSample] = []
    for record in icons["images"]:
        if not isinstance(record, dict):
            raise ValueError("dataset manifest icon records must be objects")
        value = record.get("business_id")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("dataset manifest icon record has an invalid business ID")
        card_id = str(value)
        row = crosswalk_by_id.get(card_id)
        if row is None:
            raise ValueError(f"icon record references unknown business ID: {card_id}")
        asset_variant = str(record["asset_variant"])
        if asset_variant not in row["asset_variants"]:
            raise ValueError(f"icon-to-art mapping differs from crosswalk: {record['class_name']}")
        visual_group_id = "april-fools-asari" if card_id in april_fools_ids else str(row["internal_id"])
        samples.append(
            TrainingSample(
                class_name=str(record["class_name"]),
                business_card_id=card_id,
                upgrade_count=int(row["upgrade_count"]),
                art_path=art_root / f"{asset_variant}.webp",
                visual_group_id=visual_group_id,
            )
        )
    samples.sort(key=lambda sample: natural_class_key(Path(sample.class_name)))
    sample_ids = {int(sample.business_card_id) for sample in samples}
    if sample_ids != expected_business_ids:
        raise ValueError(
            "authoritative art samples do not cover the exact declared business ID set; "
            f"missing={sorted(expected_business_ids - sample_ids)}, "
            f"surplus={sorted(sample_ids - expected_business_ids)}"
        )
    return samples


def cross_id_duplicate_groups(paths: list[Path]) -> list[dict[str, Any]]:
    """Report byte-identical source images assigned to different business IDs."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        grouped[sha256_file(path)].append(path.stem)
    collisions: list[dict[str, Any]] = []
    for image_sha256, class_names in grouped.items():
        business_ids = sorted({business_card_id(name) for name in class_names}, key=int)
        if len(business_ids) > 1:
            collisions.append(
                {
                    "image_sha256": image_sha256,
                    "business_card_ids": business_ids,
                    "class_names": sorted(class_names, key=lambda name: natural_class_key(Path(name))),
                }
            )
    return sorted(collisions, key=lambda item: int(item["business_card_ids"][0]))


def unexpected_source_collisions(collisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        collision
        for collision in collisions
        if not any(
            set(collision["business_card_ids"]).issubset(group)
            for group in KNOWN_VISUAL_AMBIGUITY_GROUPS
        )
    ]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(class_count: int) -> Any:
    import torch
    from torch import nn

    class CardEmbeddingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(128, EMBEDDING_DIM))
            self.classifier = nn.Linear(EMBEDDING_DIM, class_count, bias=False)

        def encode(self, images: Any) -> Any:
            return torch.nn.functional.normalize(self.embedding(self.encoder(images)), dim=1)

        def forward(self, images: Any) -> tuple[Any, Any]:
            vectors = self.encode(images)
            weights = torch.nn.functional.normalize(self.classifier.weight, dim=1)
            return vectors, 30.0 * vectors @ weights.T

    return CardEmbeddingNet()


class ArenaCustomizationOverlayAugment:
    """Add class-independent contest customization UI over card artwork."""

    COLORS = (
        (0, 167, 255, 238),
        (0, 91, 255, 238),
        (245, 65, 85, 238),
        (20, 206, 103, 238),
    )

    def __init__(self, probability: float) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("arena customization overlay probability must be in [0,1]")
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.probability == 0.0 or random.random() >= self.probability:
            return image
        output = image.convert("RGB").copy()
        width, height = output.size
        overlay = Image.new("RGBA", output.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        badge_count = random.randint(1, 3)
        center_x = round(width * random.uniform(0.46, 0.58))
        center_y = round(height * random.uniform(0.77, 0.88))
        radius = max(5, round(min(width, height) * random.uniform(0.105, 0.14)))
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            fill=(20, 206, 103, 242),
            outline=(255, 255, 255, 255),
            width=max(2, round(width * 0.025)),
        )
        text = str(badge_count)
        text_box = draw.textbbox((0, 0), text)
        draw.text(
            (center_x - (text_box[2] - text_box[0]) / 2, center_y - (text_box[3] - text_box[1]) / 2),
            text,
            fill=(255, 255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 110, 55, 255),
        )

        for index in range(random.randint(2, 5)):
            size = max(5, round(min(width, height) * random.uniform(0.085, 0.14)))
            x = round(width * random.uniform(0.32, 0.79))
            y = round(height * random.uniform(0.42, 0.82))
            color = random.choice(self.COLORS)
            if index % 2:
                points = ((x, y - size), (x + size, y), (x, y + size), (x - size, y))
            else:
                points = (
                    (x - size, y - size // 2),
                    (x + size // 3, y - size // 2),
                    (x + size // 3, y - size),
                    (x + size, y),
                    (x + size // 3, y + size),
                    (x + size // 3, y + size // 2),
                    (x - size, y + size // 2),
                )
            draw.polygon(points, fill=color, outline=(255, 255, 255, 255))

        return Image.alpha_composite(output.convert("RGBA"), overlay).convert("RGB")


def build_transforms(*, arena_custom_overlay_probability: float = 0.0) -> tuple[Any, Any]:
    from torchvision import transforms

    train = transforms.Compose(
        (
            ArenaCustomizationOverlayAugment(arena_custom_overlay_probability),
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.82, 1.0), ratio=(0.88, 1.12)),
            transforms.RandomAffine(degrees=3.0, translate=(0.035, 0.035), scale=(0.96, 1.04)),
            transforms.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.10, hue=0.02),
            transforms.RandomApply((transforms.GaussianBlur(3, sigma=(0.1, 0.8)),), p=0.15),
            transforms.ToTensor(),
        )
    )
    canonical = transforms.Compose((transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor()))
    return train, canonical


def load_card_art_images(paths: list[Path]) -> tuple[Image.Image, ...]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as source:
            images.append(focus_card_art(source).resize((96, 96), Image.Resampling.LANCZOS))
    return tuple(images)


def load_arena_custom_samples(
    specifications: list[str],
    samples: list[TrainingSample],
    visual_index: dict[str, int],
    *,
    weight: int,
) -> tuple[tuple[Image.Image, ...], list[int], list[dict[str, Any]]]:
    if weight < 1:
        raise ValueError("arena custom sample weight must be positive")
    group_by_business_id = {sample.business_card_id: sample.visual_group_id for sample in samples}
    images: list[Image.Image] = []
    labels: list[int] = []
    evidence: list[dict[str, Any]] = []
    for specification in specifications:
        try:
            card_id, path_text = specification.split("=", 1)
        except ValueError as error:
            raise ValueError("arena custom sample must use CARD_ID=PATH") from error
        if card_id not in group_by_business_id:
            raise ValueError(f"arena custom sample has unknown business card ID: {card_id}")
        path = Path(path_text).resolve()
        if not path.is_file():
            raise ValueError(f"arena custom sample is missing: {path}")
        with Image.open(path) as source:
            image = focus_card_art(source).resize((96, 96), Image.Resampling.LANCZOS)
        images.extend(image.copy() for _ in range(weight))
        labels.extend([visual_index[group_by_business_id[card_id]]] * weight)
        evidence.append(
            {
                "business_card_id": card_id,
                "sha256": sha256_file(path),
                "weight": weight,
            }
        )
    return tuple(images), labels, evidence


def load_upgrade_markers(samples: list[TrainingSample], icons_root: Path) -> np.ndarray:
    markers = []
    for sample in samples:
        with Image.open(icons_root / f"{sample.class_name}.webp") as source:
            markers.append(extract_upgrade_marker(source))
    return np.stack(markers).astype(np.float32)


def validate_upgrade_markers(
    samples: list[TrainingSample],
    icons_root: Path,
    templates: np.ndarray,
) -> dict[str, Any]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_group[sample.visual_group_id].append(index)
    failures = []
    query_count = 0
    for index, sample in enumerate(samples):
        candidates = by_group[sample.visual_group_id]
        states = {samples[candidate].upgrade_count for candidate in candidates}
        if states != {0, 1}:
            continue
        with Image.open(icons_root / f"{sample.class_name}.webp") as source:
            view = source.convert("RGB").resize((143, 144), Image.Resampling.BILINEAR)
            view = ImageEnhance.Brightness(view).enhance(0.92).filter(ImageFilter.GaussianBlur(0.25))
            query = extract_upgrade_marker(view)
        scores = {
            state: min(
                float(np.mean((query - templates[candidate]) ** 2))
                for candidate in candidates
                if samples[candidate].upgrade_count == state
            )
            for state in (0, 1)
        }
        predicted = int(scores[1] < scores[0])
        query_count += 1
        if predicted != sample.upgrade_count:
            failures.append(
                {
                    "class_name": sample.class_name,
                    "expected_upgrade_count": sample.upgrade_count,
                    "predicted_upgrade_count": predicted,
                    "normal_mse": scores[0],
                    "upgraded_mse": scores[1],
                }
            )
    singleton_groups = sum(len({samples[index].business_card_id for index in indices}) == 1 for indices in by_group.values())
    return {
        "upgrade_resolvable_visual_group_count": sum(
            {samples[index].upgrade_count for index in indices} == {0, 1}
            for indices in by_group.values()
        ),
        "authoritative_normal_upgraded_pair_count": sum(
            len(
                {
                    samples[index].business_card_id
                    for index in indices
                    if samples[index].upgrade_count == 0
                }
            )
            for indices in by_group.values()
            if {samples[index].upgrade_count for index in indices} == {0, 1}
        ),
        "singleton_group_count": singleton_groups,
        "query_count": query_count,
        "correct_count": query_count - len(failures),
        "accuracy": (query_count - len(failures)) / query_count,
        "failures": failures,
        "note": "separate marker ROI; excluded from embedding input and classification target",
    }


def make_dataset(images: tuple[Image.Image, ...], labels: list[int], transform: Any, *, paired: bool) -> Any:
    from torch.utils.data import Dataset

    class IconDataset(Dataset):
        def __len__(self) -> int:
            return len(images)

        def __getitem__(self, index: int) -> tuple[Any, ...]:
            image = images[index]
            first = transform(image)
            if paired:
                return first, transform(image), labels[index]
            return first, index

    return IconDataset()


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    classification_loss: float
    consistency_loss: float
    seconds: float


def train_epoch(model: Any, loader: Any, optimizer: Any, device: Any, epoch: int) -> EpochMetrics:
    import torch

    model.train()
    total_loss = total_classification = total_consistency = 0.0
    samples = 0
    started = time.perf_counter()
    for first, second, labels in loader:
        first = first.to(device, non_blocking=True)
        second = second.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        first_vectors, first_logits = model(first)
        second_vectors, second_logits = model(second)
        classification = (
            torch.nn.functional.cross_entropy(first_logits, labels)
            + torch.nn.functional.cross_entropy(second_logits, labels)
        ) / 2.0
        consistency = (1.0 - (first_vectors * second_vectors).sum(dim=1)).mean()
        loss = classification + 0.15 * consistency
        loss.backward()
        optimizer.step()
        count = labels.shape[0]
        samples += count
        total_loss += float(loss.detach()) * count
        total_classification += float(classification.detach()) * count
        total_consistency += float(consistency.detach()) * count
    return EpochMetrics(
        epoch=epoch,
        loss=total_loss / samples,
        classification_loss=total_classification / samples,
        consistency_loss=total_consistency / samples,
        seconds=time.perf_counter() - started,
    )


def encode_loader(model: Any, loader: Any, device: Any) -> np.ndarray:
    import torch

    model.eval()
    batches: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    with torch.inference_mode():
        for images, batch_indices in loader:
            batches.append(model.encode(images.to(device, non_blocking=True)).cpu().numpy())
            indices.append(batch_indices.numpy())
    vectors = np.concatenate(batches)
    order = np.argsort(np.concatenate(indices))
    return vectors[order].astype(np.float32, copy=False)


def deterministic_augmented_vectors(
    model: Any,
    images: tuple[Image.Image, ...],
    transform: Any,
    device: Any,
    seed: int,
) -> np.ndarray:
    import torch

    vectors: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for index, image in enumerate(images):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + index * 104729)
                tensor = transform(image)[None, ...].to(device)
                vectors.append(model.encode(tensor).cpu().numpy()[0])
    return np.stack(vectors).astype(np.float32, copy=False)


def topk_visual_metrics(
    query: np.ndarray,
    gallery: np.ndarray,
    class_names: list[str],
    visual_group_ids: list[str],
) -> dict[str, Any]:
    source_ids = np.asarray([business_card_id(name) for name in class_names])
    source_groups = np.asarray(visual_group_ids)
    groups = np.asarray(sorted(set(source_groups)))
    similarities = query @ gallery.T
    group_similarities = np.stack(
        [similarities[:, source_groups == group].max(axis=1) for group in groups],
        axis=1,
    )
    top_k = min(10, group_similarities.shape[1])
    top_five = np.argpartition(-group_similarities, kth=top_k - 1, axis=1)[:, :top_k]
    top_five = np.take_along_axis(
        top_five,
        np.argsort(-np.take_along_axis(group_similarities, top_five, axis=1), axis=1),
        axis=1,
    )
    predicted_groups = groups[top_five]
    correct = source_groups[:, None] == predicted_groups
    top1 = float(correct[:, 0].mean())
    top5 = float(correct[:, :5].any(axis=1).mean())
    top_two = np.sort(np.partition(group_similarities, kth=-2, axis=1)[:, -2:], axis=1)
    margins = top_two[:, 1] - top_two[:, 0]
    top1_scores = np.take_along_axis(group_similarities, top_five[:, :1], axis=1)
    failure_indices = np.flatnonzero(~correct[:, 0])
    failures = [
        {
            "class_name": class_names[int(index)],
            "expected_card_id": str(source_ids[index]),
            "expected_visual_group_id": str(source_groups[index]),
            "predicted_visual_group_id": str(predicted_groups[index, 0]),
            "top10_visual_group_ids": [str(value) for value in predicted_groups[index]],
            "top1_similarity": float(top1_scores[index, 0]),
            "margin": float(margins[index]),
        }
        for index in failure_indices
    ]
    return {
        "query_count": int(query.shape[0]),
        "top1_visual_identity_accuracy": top1,
        "top5_visual_identity_accuracy": top5,
        "top10_visual_identity_accuracy": float(correct.any(axis=1).mean()),
        "wrong_visual_identity_count": int((~correct[:, 0]).sum()),
        "raw_retrieval_unknown_count": 0,
        "top1_similarity_min": float(top1_scores.min()),
        "top1_similarity_median": float(np.median(top1_scores)),
        "visual_identity_margin_median": float(np.median(margins)),
        "failures": failures,
        "visual_identity_count": len(groups),
        "note": "deterministic augmentation gate over card artwork only; upgrade and plan resolve exact ID after retrieval",
    }


def export_onnx(model: Any, destination: Path, device: Any) -> None:
    import torch
    from torch import nn

    class EncoderOnly(nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.source = source

        def forward(self, images: Any) -> Any:
            return self.source.encode(images)

    wrapper = EncoderOnly(model).eval().to(device)
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    torch.onnx.export(
        wrapper,
        dummy,
        destination,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the 128-d card embedding and gallery")
    parser.add_argument("--art-root", type=Path, required=True)
    parser.add_argument("--icons-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=750418)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--arena-custom-overlay-probability", type=float, default=0.0)
    parser.add_argument("--arena-custom-sample", action="append", default=[])
    parser.add_argument("--arena-custom-sample-weight", type=int, default=64)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    if args.epochs < 1 or args.batch_size < 2:
        parser.error("epochs must be >=1 and batch-size must be >=2")
    if not 0.0 <= args.arena_custom_overlay_probability <= 1.0:
        parser.error("--arena-custom-overlay-probability must be in [0,1]")
    if args.arena_custom_sample_weight < 1:
        parser.error("--arena-custom-sample-weight must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    art_root = args.art_root.resolve()
    icons_root = args.icons_root.resolve()
    dataset_manifest_path = args.dataset_manifest.resolve()
    dataset_manifest = verify_dataset_manifest(dataset_manifest_path, icons_root)
    dataset_manifest_sha256 = sha256_file(dataset_manifest_path)
    samples = discover_art_samples(dataset_manifest, art_root)
    paths = [sample.art_path for sample in samples]
    images = load_card_art_images(paths)
    class_names = [sample.class_name for sample in samples]
    visual_group_ids = [sample.visual_group_id for sample in samples]
    business_ids = sorted({business_card_id(name) for name in class_names}, key=int)
    visual_groups = sorted(set(visual_group_ids))
    visual_index = {group: index for index, group in enumerate(visual_groups)}
    training_labels = [visual_index[group] for group in visual_group_ids]
    arena_images, arena_labels, arena_sample_evidence = load_arena_custom_samples(
        args.arena_custom_sample,
        samples,
        visual_index,
        weight=args.arena_custom_sample_weight,
    )
    seed_everything(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    train_transform, canonical_transform = build_transforms(
        arena_custom_overlay_probability=args.arena_custom_overlay_probability
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        make_dataset(
            images + arena_images,
            training_labels + arena_labels,
            train_transform,
            paired=True,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    canonical_loader = DataLoader(
        make_dataset(images, training_labels, canonical_transform, paired=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(len(visual_groups)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    start_epoch = 1
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["class_names"] != class_names or checkpoint.get("business_ids") != business_ids:
            raise ValueError("resume checkpoint class table differs from the pinned source")
        if checkpoint.get("visual_group_ids") != visual_group_ids:
            raise ValueError("resume checkpoint visual-group table differs from authoritative art")
        if checkpoint.get("known_visual_ambiguity_groups") != [
            sorted(group, key=int) for group in KNOWN_VISUAL_AMBIGUITY_GROUPS
        ]:
            raise ValueError("resume checkpoint visual ambiguity contract is missing or differs")
        if checkpoint.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            raise ValueError("resume checkpoint dataset manifest differs from the current source")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))

    checkpoint_path = args.output_dir / "checkpoint.pt"
    for epoch in range(start_epoch, args.epochs + 1):
        metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        history.append(asdict(metrics))
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "epoch": epoch,
                "class_names": class_names,
                "business_ids": business_ids,
                "visual_group_ids": visual_group_ids,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "seed": args.seed,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "known_visual_ambiguity_groups": [
                    sorted(group, key=int) for group in KNOWN_VISUAL_AMBIGUITY_GROUPS
                ],
            },
            checkpoint_path,
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    gallery_vectors = encode_loader(model, canonical_loader, device)
    validation_vectors = deterministic_augmented_vectors(model, images, train_transform, device, args.seed + 10_000_000)
    validation_metrics = topk_visual_metrics(validation_vectors, gallery_vectors, class_names, visual_group_ids)
    clean_transform, _ = build_transforms(arena_custom_overlay_probability=0.0)
    clean_validation_vectors = deterministic_augmented_vectors(
        model,
        images,
        clean_transform,
        device,
        args.seed + 20_000_000,
    )
    clean_validation_metrics = topk_visual_metrics(
        clean_validation_vectors,
        gallery_vectors,
        class_names,
        visual_group_ids,
    )

    model_path = args.output_dir / "card_embedding_model.onnx"
    gallery_path = args.output_dir / "card_embedding_gallery.npz"
    classes_path = args.output_dir / "card_embedding_classes.json"
    metrics_path = args.output_dir / "augmentation_validation.json"
    clean_metrics_path = args.output_dir / "clean_augmentation_validation.json"
    manifest_path = args.output_dir / "manifest.json"
    export_onnx(model, model_path, device)
    upgrade_markers = load_upgrade_markers(samples, icons_root)
    validation_metrics["upgrade_state"] = validate_upgrade_markers(samples, icons_root, upgrade_markers)
    clean_validation_metrics["upgrade_state"] = validation_metrics["upgrade_state"]
    np.savez_compressed(
        gallery_path,
        embeddings=gallery_vectors,
        class_names=np.asarray(class_names),
        card_ids=np.asarray([business_card_id(name) for name in class_names]),
        visual_group_ids=np.asarray(visual_group_ids),
        upgrade_counts=np.asarray([sample.upgrade_count for sample in samples], dtype=np.int8),
        upgrade_markers=upgrade_markers,
    )
    write_json(classes_path, class_names)
    write_json(metrics_path, validation_metrics)
    write_json(clean_metrics_path, clean_validation_metrics)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_seconds": int(time.time()),
        "source": {
            "dataset_id": dataset_manifest.get("dataset_id"),
            "dataset_revision": dataset_manifest.get("dataset_revision"),
            "dataset_manifest": str(dataset_manifest_path),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "art_root": str(art_root),
            "upgrade_marker_source": str(icons_root),
            "image_class_count": len(class_names),
            "visual_identity_count": len(visual_groups),
            **business_id_manifest_fields([int(value) for value in business_ids]),
            "known_visual_ambiguity_groups": [
                sorted(group, key=int) for group in KNOWN_VISUAL_AMBIGUITY_GROUPS
            ],
            "recognition_source": "official raw card art only; score/cost and side-function UI excluded",
            "card_art_exclusion_boxes": CARD_ART_EXCLUSION_BOXES,
            "exact_id_resolution": "UPGRADE_STATE; APRIL_FOOLS_GROUP_ALSO_REQUIRES_PLAN",
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "history": history,
            "classification_target": f"{len(visual_groups)}_card_art_visual_identities",
            "arena_custom_overlay_probability": args.arena_custom_overlay_probability,
            "arena_custom_samples": arena_sample_evidence,
        },
        "model": {
            "path": model_path.name,
            "sha256": sha256_file(model_path),
            "input": "float32[N,3,64,64] RGB range [0,1]",
            "output": "embedding float32[N,128] L2-normalised",
        },
        "gallery": {
            "path": gallery_path.name,
            "sha256": sha256_file(gallery_path),
            "embedding_dim": EMBEDDING_DIM,
            "distance": "cosine_similarity",
            "class_table_path": classes_path.name,
            "class_table_sha256": sha256_file(classes_path),
        },
        "validation": {
            "path": metrics_path.name,
            "sha256": sha256_file(metrics_path),
            "clean_path": clean_metrics_path.name,
            "clean_sha256": sha256_file(clean_metrics_path),
            "independent_dmm_atlas_gate": "PENDING",
            "acceptance_calibration": "PENDING",
        },
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "metrics": {key: value for key, value in validation_metrics.items() if key != "failures"},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
