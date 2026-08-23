from __future__ import annotations

import sys
import json
import time
import random
import argparse
import importlib.util
from typing import Any
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.p_item_recognition.preprocess import (
    focus_p_item_art,
    extract_p_item_upgrade_marker,
)

CARD_TRAINER_PATH = PROJECT_ROOT / "tools" / "card-embedding" / "train_card_embedding.py"
_SPEC = importlib.util.spec_from_file_location("task085_shared_embedding_training", CARD_TRAINER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load shared card-embedding training primitives")
_SHARED = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SHARED
_SPEC.loader.exec_module(_SHARED)

EMBEDDING_DIM = 128
INPUT_SIZE = 64
SCHEMA_VERSION = 1
EXPECTED_BUSINESS_ID_COUNT = 420
EXPECTED_VISUAL_IDENTITY_COUNT = 273
KNOWN_PLAN_AMBIGUITY = {
    "business_ids": (406, 407, 408),
    "plan_to_business_id": {"sense": 406, "logic": 407, "anomaly": 408},
}
GALLERY_AUGMENTATION_SEED_OFFSETS = (30_000_000, 40_000_000, 50_000_000, 60_000_000)
LOW_RESOLUTION_PROBABILITY = 0.80
LOW_RESOLUTION_LOWER_EDGE = 16
LOW_RESOLUTION_SPLIT_EDGE = 40
GALLERY_LOW_RESOLUTION_EDGES = (16, 24, 32, 48)


sha256_file = _SHARED.sha256_file
seed_everything = _SHARED.seed_everything
build_model = _SHARED.build_model
make_dataset = _SHARED.make_dataset
train_epoch = _SHARED.train_epoch
encode_loader = _SHARED.encode_loader
export_onnx = _SHARED.export_onnx
write_json = _SHARED.write_json


@dataclass(frozen=True)
class PItemSample:
    class_name: str
    p_item_id: str
    upgraded: int
    plan: str
    art_path: Path
    reference_path: Path
    visual_group_id: str


def verify_dataset_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("validation", {}).get("status") != "CLEAR":
        raise ValueError("authoritative P-item dataset validation must be CLEAR")
    scope = manifest.get("scope", {})
    business_ids = [int(value) for value in scope.get("business_ids", [])]
    if (
        scope.get("rule") != "sourceType != produce"
        or len(business_ids) != EXPECTED_BUSINESS_ID_COUNT
        or len(set(business_ids)) != EXPECTED_BUSINESS_ID_COUNT
    ):
        raise ValueError("P-item dataset scope must contain 420 unique non-produce business IDs")
    if manifest.get("validation", {}).get("visual_identity_count") != EXPECTED_VISUAL_IDENTITY_COUNT:
        raise ValueError("P-item dataset must contain exactly 273 visual identities")
    return manifest


def _verify_files(root: Path, records: list[dict[str, Any]], pattern: str) -> None:
    expected = {str(record["path"]): str(record["sha256"]).upper() for record in records}
    actual = {path.name: sha256_file(path) for path in root.glob(pattern)}
    if not expected or actual != expected:
        raise ValueError(f"dataset files or SHA-256 values differ under {root}")


def discover_samples(manifest: dict[str, Any], manifest_path: Path) -> list[PItemSample]:
    dataset_root = manifest_path.parent
    art_root = dataset_root / str(manifest["official_art"]["root"])
    reference_root = dataset_root / str(manifest["reference_icons"]["root"])
    _verify_files(art_root, list(manifest["official_art"]["images"]), "*.webp")
    _verify_files(reference_root, list(manifest["reference_icons"]["images"]), "*.png")
    crosswalk_path = dataset_root / str(manifest["mapping"]["crosswalk_path"])
    if sha256_file(crosswalk_path) != str(manifest["mapping"]["crosswalk_sha256"]).upper():
        raise ValueError("P-item crosswalk SHA-256 differs from the dataset manifest")
    rows = json.loads(crosswalk_path.read_text(encoding="utf-8"))
    samples = [
        PItemSample(
            class_name=str(row["business_id"]),
            p_item_id=str(row["business_id"]),
            upgraded=int(bool(row["upgraded"])),
            plan=str(row["plan"]),
            art_path=art_root / f"{row['asset_name']}.webp",
            reference_path=reference_root / f"{row['business_id']}.png",
            visual_group_id=str(row["asset_name"]),
        )
        for row in rows
    ]
    samples.sort(key=lambda sample: int(sample.p_item_id))
    expected_ids = {str(value) for value in manifest["scope"]["business_ids"]}
    if {sample.p_item_id for sample in samples} != expected_ids or len(samples) != len(expected_ids):
        raise ValueError("P-item samples do not cover the frozen business-ID scope exactly once")
    if len({sample.visual_group_id for sample in samples}) != EXPECTED_VISUAL_IDENTITY_COUNT:
        raise ValueError("P-item samples do not contain exactly 273 visual identities")
    ambiguity = [
        sample
        for sample in samples
        if int(sample.p_item_id) in KNOWN_PLAN_AMBIGUITY["business_ids"]
    ]
    if (
        len({sample.visual_group_id for sample in ambiguity}) != 1
        or {sample.plan: int(sample.p_item_id) for sample in ambiguity}
        != KNOWN_PLAN_AMBIGUITY["plan_to_business_id"]
    ):
        raise ValueError("known 406/407/408 plan ambiguity contract differs from the dataset")
    return samples


def load_art_images(samples: list[PItemSample]) -> tuple[Image.Image, ...]:
    images = []
    for sample in samples:
        with Image.open(sample.art_path) as source:
            images.append(source.convert("RGBA").resize((108, 108), Image.Resampling.LANCZOS))
    return tuple(images)


def load_reference_images(samples: list[PItemSample]) -> tuple[Image.Image, ...]:
    images = []
    for sample in samples:
        with Image.open(sample.reference_path) as source:
            images.append(focus_p_item_art(source))
    return tuple(images)


class RandomBackground:
    def __call__(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        if random.random() < 0.18:
            colour = (255, 255, 255, 255)
        else:
            colour = tuple(random.randint(low, 255) for low in (224, 224, 224)) + (255,)
        background = Image.new("RGBA", rgba.size, colour)
        background.alpha_composite(rgba)
        return background.convert("RGB")


class WhiteBackground:
    def __call__(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")


class RandomLowResolution:
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= LOW_RESOLUTION_PROBABILITY:
            return image
        edge = (
            random.randint(LOW_RESOLUTION_LOWER_EDGE, LOW_RESOLUTION_SPLIT_EDGE)
            if random.random() < 0.60
            else random.randint(LOW_RESOLUTION_SPLIT_EDGE + 1, INPUT_SIZE)
        )
        return image.resize((edge, edge), Image.Resampling.BILINEAR).resize(
            (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR
        )


class RandomEdgeInterference:
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= 0.35:
            return image
        output = image.copy()
        draw = ImageDraw.Draw(output)
        colour = tuple(random.randint(190, 255) for _ in range(3))
        edge = random.choice(("top", "bottom", "left", "right"))
        thickness = random.randint(1, 6)
        if edge == "top":
            box = (0, 0, INPUT_SIZE, thickness)
        elif edge == "bottom":
            box = (0, INPUT_SIZE - thickness, INPUT_SIZE, INPUT_SIZE)
        elif edge == "left":
            box = (0, 0, thickness, INPUT_SIZE)
        else:
            box = (INPUT_SIZE - thickness, 0, INPUT_SIZE, INPUT_SIZE)
        draw.rectangle(box, fill=colour)
        return output


def build_transforms() -> tuple[Any, Any]:
    from torchvision import transforms

    train = transforms.Compose(
        (
            RandomBackground(),
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.78, 1.0), ratio=(0.88, 1.12)),
            transforms.RandomAffine(degrees=4.0, translate=(0.05, 0.05), scale=(0.94, 1.06)),
            transforms.ColorJitter(brightness=0.24, contrast=0.18, saturation=0.12, hue=0.025),
            transforms.RandomApply((transforms.GaussianBlur(3, sigma=(0.1, 0.9)),), p=0.20),
            RandomLowResolution(),
            RandomEdgeInterference(),
            transforms.ToTensor(),
        )
    )
    canonical = transforms.Compose(
        (WhiteBackground(), transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor())
    )
    return train, canonical


def build_reference_transform() -> Any:
    from torchvision import transforms

    return transforms.Compose((transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor()))


def build_gallery_low_resolution_transform(edge: int, *, transparent_art: bool) -> Any:
    from torchvision import transforms

    operations: list[Any] = []
    if transparent_art:
        operations.append(WhiteBackground())
    operations.extend(
        (
            transforms.Resize((edge, edge)),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
        )
    )
    return transforms.Compose(tuple(operations))


def deterministic_augmented_vectors(
    model: Any,
    images: tuple[Image.Image, ...],
    transform: Any,
    device: Any,
    seed: int,
) -> np.ndarray:
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    vectors: list[np.ndarray] = []
    model.eval()
    try:
        with torch.inference_mode():
            for index, image in enumerate(images):
                sample_seed = seed + index * 104729
                random.seed(sample_seed)
                np.random.seed(sample_seed % (2**32))
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(sample_seed)
                    tensor = transform(image)[None, ...].to(device)
                    vectors.append(model.encode(tensor).cpu().numpy()[0])
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return np.stack(vectors).astype(np.float32, copy=False)


def topk_visual_metrics(
    query: np.ndarray,
    gallery: np.ndarray,
    samples: list[PItemSample],
    gallery_visual_group_ids: list[str] | None = None,
) -> dict[str, Any]:
    source_groups = np.asarray([sample.visual_group_id for sample in samples])
    gallery_groups = np.asarray(gallery_visual_group_ids or source_groups.tolist())
    if len(gallery_groups) != len(gallery):
        raise ValueError("gallery vectors and visual-group metadata have different lengths")
    groups = np.asarray(sorted(set(source_groups)))
    similarities = query @ gallery.T
    group_similarities = np.stack(
        [similarities[:, gallery_groups == group].max(axis=1) for group in groups], axis=1
    )
    top_k = min(10, len(groups))
    candidates = np.argpartition(-group_similarities, kth=top_k - 1, axis=1)[:, :top_k]
    candidates = np.take_along_axis(
        candidates,
        np.argsort(-np.take_along_axis(group_similarities, candidates, axis=1), axis=1),
        axis=1,
    )
    predicted_groups = groups[candidates]
    correct = source_groups[:, None] == predicted_groups
    top_scores = np.take_along_axis(group_similarities, candidates[:, :1], axis=1)[:, 0]
    top_two = np.sort(np.partition(group_similarities, kth=-2, axis=1)[:, -2:], axis=1)
    margins = top_two[:, 1] - top_two[:, 0]
    failures = [
        {
            "p_item_id": samples[int(index)].p_item_id,
            "expected_visual_group_id": str(source_groups[index]),
            "predicted_visual_group_id": str(predicted_groups[index, 0]),
            "top10_visual_group_ids": [str(value) for value in predicted_groups[index]],
            "top1_similarity": float(top_scores[index]),
            "margin": float(margins[index]),
        }
        for index in np.flatnonzero(~correct[:, 0])
    ]
    return {
        "query_count": len(samples),
        "visual_identity_count": len(groups),
        "top1_visual_identity_accuracy": float(correct[:, 0].mean()),
        "top5_visual_identity_accuracy": float(correct[:, :5].any(axis=1).mean()),
        "top10_visual_identity_accuracy": float(correct.any(axis=1).mean()),
        "wrong_visual_identity_count": len(failures),
        "top1_similarity_min": float(top_scores.min()),
        "top1_similarity_median": float(np.median(top_scores)),
        "visual_identity_margin_min": float(margins.min()),
        "visual_identity_margin_median": float(np.median(margins)),
        "failures": failures,
    }


def load_upgrade_markers(samples: list[PItemSample]) -> np.ndarray:
    markers = []
    for sample in samples:
        with Image.open(sample.reference_path) as source:
            markers.append(extract_p_item_upgrade_marker(source))
    return np.stack(markers).astype(np.float32)


def validate_upgrade_markers(
    samples: list[PItemSample], templates: np.ndarray
) -> dict[str, Any]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_group[sample.visual_group_id].append(index)
    failures = []
    score_rows = []
    for index, sample in enumerate(samples):
        candidates = by_group[sample.visual_group_id]
        if {samples[candidate].upgraded for candidate in candidates} != {0, 1}:
            continue
        with Image.open(sample.reference_path) as source:
            view = source.convert("RGB")
            view = ImageEnhance.Brightness(view).enhance(0.92).filter(
                ImageFilter.GaussianBlur(0.25)
            )
            query = extract_p_item_upgrade_marker(view)
        scores = {
            state: min(
                float(np.mean((query - templates[candidate]) ** 2))
                for candidate in candidates
                if samples[candidate].upgraded == state
            )
            for state in (0, 1)
        }
        predicted = int(scores[1] < scores[0])
        row = {
            "p_item_id": sample.p_item_id,
            "expected_upgrade_state": sample.upgraded,
            "predicted_upgrade_state": predicted,
            "best_mse": scores[predicted],
            "margin": scores[1 - predicted] - scores[predicted],
        }
        score_rows.append(row)
        if predicted != sample.upgraded:
            failures.append(row)
    return {
        "pair_count": sum(
            {samples[index].upgraded for index in indices} == {0, 1}
            for indices in by_group.values()
        ),
        "query_count": len(score_rows),
        "correct_count": len(score_rows) - len(failures),
        "accuracy": (len(score_rows) - len(failures)) / len(score_rows),
        "maximum_correct_distance": max(row["best_mse"] for row in score_rows),
        "minimum_correct_margin": min(row["margin"] for row in score_rows),
        "failures": failures,
        "note": "top-right + marker is excluded from the embedding and resolved independently",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the P-item embedding and gallery")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=850418)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    if args.epochs < 1 or args.batch_size < 2:
        parser.error("epochs must be >=1 and batch-size must be >=2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dataset_manifest.resolve()
    dataset_manifest = verify_dataset_manifest(manifest_path)
    dataset_manifest_sha256 = sha256_file(manifest_path)
    samples = discover_samples(dataset_manifest, manifest_path)
    art_images = load_art_images(samples)
    reference_images = load_reference_images(samples)
    class_names = [sample.class_name for sample in samples]
    p_item_ids = [sample.p_item_id for sample in samples]
    visual_group_ids = [sample.visual_group_id for sample in samples]
    visual_groups = sorted(set(visual_group_ids))
    visual_index = {group: index for index, group in enumerate(visual_groups)}
    training_labels = [visual_index[group] for group in visual_group_ids]

    seed_everything(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    train_transform, canonical_transform = build_transforms()
    reference_transform = build_reference_transform()
    generator = torch.Generator().manual_seed(args.seed)
    mixed_training_images = art_images + reference_images
    mixed_training_labels = training_labels + training_labels
    train_loader = DataLoader(
        make_dataset(mixed_training_images, mixed_training_labels, train_transform, paired=True),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    canonical_loader = DataLoader(
        make_dataset(art_images, training_labels, canonical_transform, paired=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    reference_loader = DataLoader(
        make_dataset(reference_images, training_labels, reference_transform, paired=False),
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
        if checkpoint.get("class_names") != class_names:
            raise ValueError("resume checkpoint class table differs from the pinned P-item source")
        if checkpoint.get("visual_group_ids") != visual_group_ids:
            raise ValueError("resume checkpoint visual-group table differs")
        if checkpoint.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            raise ValueError("resume checkpoint dataset manifest differs")
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
                "p_item_ids": p_item_ids,
                "visual_group_ids": visual_group_ids,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "seed": args.seed,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "known_plan_ambiguity": KNOWN_PLAN_AMBIGUITY,
            },
            checkpoint_path,
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    official_gallery_vectors = encode_loader(model, canonical_loader, device)
    reference_gallery_vectors = encode_loader(model, reference_loader, device)
    gallery_batches = [official_gallery_vectors, reference_gallery_vectors]
    gallery_variant_names = ["official", "reference"]
    for variant_index, seed_offset in enumerate(GALLERY_AUGMENTATION_SEED_OFFSETS, start=1):
        gallery_batches.append(
            deterministic_augmented_vectors(
                model, art_images, train_transform, device, args.seed + seed_offset
            )
        )
        gallery_variant_names.append(f"official_aug{variant_index}")
        gallery_batches.append(
            deterministic_augmented_vectors(
                model,
                reference_images,
                train_transform,
                device,
                args.seed + seed_offset + 5_000_000,
            )
        )
        gallery_variant_names.append(f"reference_aug{variant_index}")
    for edge in GALLERY_LOW_RESOLUTION_EDGES:
        gallery_batches.append(
            deterministic_augmented_vectors(
                model,
                art_images,
                build_gallery_low_resolution_transform(edge, transparent_art=True),
                device,
                args.seed + edge * 100_000,
            )
        )
        gallery_variant_names.append(f"official_low{edge}")
        gallery_batches.append(
            deterministic_augmented_vectors(
                model,
                reference_images,
                build_gallery_low_resolution_transform(edge, transparent_art=False),
                device,
                args.seed + edge * 100_000 + 50_000,
            )
        )
        gallery_variant_names.append(f"reference_low{edge}")
    gallery_vectors = np.concatenate(gallery_batches, axis=0).astype(np.float32, copy=False)
    gallery_visual_group_ids = visual_group_ids * len(gallery_variant_names)
    gallery_class_names = [
        f"{class_name}_{variant_name}"
        for variant_name in gallery_variant_names
        for class_name in class_names
    ]
    gallery_p_item_ids = p_item_ids * len(gallery_variant_names)
    augmentation_vectors = deterministic_augmented_vectors(
        model, art_images, train_transform, device, args.seed + 10_000_000
    )
    reference_vectors = deterministic_augmented_vectors(
        model, reference_images, train_transform, device, args.seed + 20_000_000
    )
    validation_metrics = {
        "official_art_augmentation": topk_visual_metrics(
            augmentation_vectors,
            gallery_vectors,
            samples,
            gallery_visual_group_ids,
        ),
        "fixed_repository_icon_perturbation": topk_visual_metrics(
            reference_vectors,
            gallery_vectors,
            samples,
            gallery_visual_group_ids,
        ),
        "fixed_repository_icons_against_official_only_gallery": topk_visual_metrics(
            reference_gallery_vectors, official_gallery_vectors, samples
        ),
    }

    model_path = args.output_dir / "p_item_embedding_model.onnx"
    gallery_path = args.output_dir / "p_item_embedding_gallery.npz"
    classes_path = args.output_dir / "p_item_embedding_classes.json"
    metrics_path = args.output_dir / "augmentation_validation.json"
    output_manifest_path = args.output_dir / "manifest.json"
    export_onnx(model, model_path, device)
    upgrade_markers = load_upgrade_markers(samples)
    validation_metrics["upgrade_state"] = validate_upgrade_markers(samples, upgrade_markers)
    gallery_upgrade_markers = np.concatenate(
        [upgrade_markers] * len(gallery_variant_names), axis=0
    )
    gallery_upgrade_states = np.asarray(
        [sample.upgraded for sample in samples] * len(gallery_variant_names), dtype=np.int8
    )
    gallery_plans = np.asarray([sample.plan for sample in samples] * len(gallery_variant_names))
    np.savez_compressed(
        gallery_path,
        embeddings=gallery_vectors,
        class_names=np.asarray(gallery_class_names),
        p_item_ids=np.asarray(gallery_p_item_ids),
        visual_group_ids=np.asarray(gallery_visual_group_ids),
        upgrade_states=gallery_upgrade_states,
        upgrade_markers=gallery_upgrade_markers,
        plans=gallery_plans,
    )
    write_json(classes_path, gallery_class_names)
    write_json(metrics_path, validation_metrics)
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_seconds": int(time.time()),
        "source": {
            "dataset_id": dataset_manifest["dataset_id"],
            "dataset_revision": dataset_manifest["dataset_revision"],
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "business_id_count": len(samples),
            "business_ids": [int(value) for value in p_item_ids],
            "visual_identity_count": len(visual_groups),
            "scope": "sourceType != produce",
            "known_plan_ambiguity": KNOWN_PLAN_AMBIGUITY,
            "recognition_source": (
                "official transparent P-item art plus fixed-revision rendered icons after exact "
                "official-art crosswalk verification; frame, background and + marker excluded"
            ),
            "exact_id_resolution": "UPGRADE_STATE; 406/407/408 ALSO_REQUIRE_PLAN",
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
            "classification_target": "273_p_item_visual_identities",
            "training_domain_count": 2,
            "training_sample_count": len(mixed_training_images),
            "augmentation": "small-icon, low-resolution, scale, compression-like blur, pastel background, edge interference",
            "low_resolution_probability": LOW_RESOLUTION_PROBABILITY,
            "low_resolution_edge_range": [LOW_RESOLUTION_LOWER_EDGE, INPUT_SIZE],
            "gallery_low_resolution_edges": list(GALLERY_LOW_RESOLUTION_EDGES),
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
            "top_k": 5,
            "variant_count": len(gallery_class_names),
            "variants_per_business_id": len(gallery_variant_names),
            "variant_names": gallery_variant_names,
            "class_table_path": classes_path.name,
            "class_table_sha256": sha256_file(classes_path),
        },
        "validation": {
            "path": metrics_path.name,
            "sha256": sha256_file(metrics_path),
            "acceptance_calibration": "PENDING",
        },
    }
    write_json(output_manifest_path, output_manifest)
    print(
        json.dumps(
            {
                "manifest": str(output_manifest_path),
                "official_art_top1": validation_metrics["official_art_augmentation"][
                    "top1_visual_identity_accuracy"
                ],
                "reference_icon_top1": validation_metrics[
                    "fixed_repository_icon_perturbation"
                ]["top1_visual_identity_accuracy"],
                "reference_icon_top5": validation_metrics[
                    "fixed_repository_icon_perturbation"
                ]["top5_visual_identity_accuracy"],
                "upgrade_accuracy": validation_metrics["upgrade_state"]["accuracy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
