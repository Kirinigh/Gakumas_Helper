from __future__ import annotations

import io
import sys
import json
import time
import argparse
from typing import Any
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.p_item_recognition.embedding import (
    OnnxPItemEmbedder,
    PItemEmbeddingGallery,
    PItemEmbeddingRecognizer,
)
from agent.p_item_recognition.preprocess import extract_p_item_upgrade_marker


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalise_icon(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize((130, 130), Image.Resampling.LANCZOS)


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def perturbations(image: Image.Image) -> dict[str, Image.Image]:
    source = normalise_icon(image)
    variants = {"canonical": source}
    for edge in (64, 56, 40, 32, 28, 20, 16):
        variants[f"low_resolution_{edge}"] = source.resize(
            (edge, edge), Image.Resampling.BILINEAR
        ).resize((130, 130), Image.Resampling.BILINEAR)
    variants["jpeg_quality_40"] = jpeg_roundtrip(source, 40)
    variants["gaussian_blur_0_8"] = source.filter(ImageFilter.GaussianBlur(0.8))
    variants["brightness_0_75"] = ImageEnhance.Brightness(source).enhance(0.75)
    variants["centre_crop_6pct"] = source.crop((8, 8, 122, 122)).resize(
        (130, 130), Image.Resampling.BILINEAR
    )
    edge_ui = source.copy()
    draw = ImageDraw.Draw(edge_ui)
    draw.rectangle((0, 0, 7, 129), fill=(240, 220, 255))
    draw.rectangle((0, 122, 129, 129), fill=(220, 245, 255))
    variants["edge_ui_interference"] = edge_ui
    return variants


def marker_feature(marker: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(marker, dtype=np.float32)
    if mode == "raw_mse":
        return values
    if mode == "global_cosine":
        flat = values.reshape(-1)
        norm = float(np.linalg.norm(flat))
        return flat / max(norm, 1e-8)
    if mode == "global_zscore":
        return (values - float(values.mean())) / max(float(values.std()), 1e-6)
    if mode == "channel_zscore":
        mean = values.mean(axis=(0, 1), keepdims=True)
        deviation = values.std(axis=(0, 1), keepdims=True)
        return (values - mean) / np.maximum(deviation, 1e-6)
    raise ValueError(f"unsupported marker metric: {mode}")


def evaluate_upgrade_marker_metrics(
    rows: list[dict[str, Any]], reference_root: Path
) -> dict[str, Any]:
    group_indices: dict[str, list[int]] = {}
    canonical_markers: list[np.ndarray] = []
    image_variants: list[dict[str, Image.Image]] = []
    for index, row in enumerate(rows):
        group_indices.setdefault(str(row["asset_name"]), []).append(index)
        with Image.open(reference_root / f"{row['business_id']}.png") as source:
            canonical_markers.append(extract_p_item_upgrade_marker(source))
            image_variants.append(perturbations(source))

    result: dict[str, Any] = {}
    for mode in ("raw_mse", "global_cosine", "global_zscore", "channel_zscore"):
        per_perturbation: dict[str, Any] = {}
        for perturbation in image_variants[0]:
            records = []
            for index, row in enumerate(rows):
                candidates = group_indices[str(row["asset_name"])]
                if {int(bool(rows[candidate]["upgraded"])) for candidate in candidates} != {
                    0,
                    1,
                }:
                    continue
                query = marker_feature(
                    extract_p_item_upgrade_marker(image_variants[index][perturbation]), mode
                )
                scores = {
                    state: min(
                        float(
                            np.mean(
                                (
                                    query
                                    - marker_feature(canonical_markers[candidate], mode)
                                )
                                ** 2
                            )
                        )
                        for candidate in candidates
                        if int(bool(rows[candidate]["upgraded"])) == state
                    )
                    for state in (0, 1)
                }
                predicted = int(scores[1] < scores[0])
                expected = int(bool(row["upgraded"]))
                records.append(
                    {
                        "p_item_id": int(row["business_id"]),
                        "expected": expected,
                        "predicted": predicted,
                        "correct": predicted == expected,
                        "best_distance": scores[predicted],
                        "margin": scores[1 - predicted] - scores[predicted],
                    }
                )
            correct = [record for record in records if record["correct"]]
            per_perturbation[perturbation] = {
                "query_count": len(records),
                "accuracy": sum(record["correct"] for record in records) / len(records),
                "maximum_correct_distance": max(
                    record["best_distance"] for record in correct
                ),
                "minimum_correct_margin": min(record["margin"] for record in correct),
                "wrong_count": len(records) - len(correct),
                "failures": [record for record in records if not record["correct"]],
            }
        result[mode] = per_perturbation
    return result


def negative_images(seed: int, count: int) -> list[tuple[str, Image.Image]]:
    rng = np.random.default_rng(seed)
    images: list[tuple[str, Image.Image]] = []
    colours = ((255, 255, 255), (0, 0, 0), (240, 220, 235), (40, 80, 120))
    for index, colour in enumerate(colours):
        images.append((f"solid_{index}", Image.new("RGB", (130, 130), colour)))
    while len(images) < count:
        index = len(images)
        mode = index % 3
        if mode == 0:
            values = rng.integers(0, 256, (130, 130, 3), dtype=np.uint8)
        elif mode == 1:
            x = np.linspace(0, 255, 130, dtype=np.uint8)
            values = np.repeat(x[None, :, None], 130, axis=0)
            values = np.repeat(values, 3, axis=2)
            values = np.roll(values, index % 130, axis=0)
        else:
            values = np.full((130, 130, 3), rng.integers(190, 256, 3), dtype=np.uint8)
            for _ in range(8):
                x0, y0 = rng.integers(0, 110, 2)
                width, height = rng.integers(4, 30, 2)
                values[y0 : y0 + height, x0 : x0 + width] = rng.integers(0, 180, 3)
        images.append((f"synthetic_unknown_{index}", Image.fromarray(values, mode="RGB")))
    return images


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def calibrate_thresholds(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    *,
    target_field: str,
) -> dict[str, Any]:
    def is_wrong(record: dict[str, Any]) -> bool:
        if record[target_field]:
            return False
        if target_field == "exact_correct":
            return record["predicted_p_item_id"] != "UNKNOWN"
        return True

    score_values = sorted(
        {
            round(float(record["similarity"]), 4)
            for record in positives + negatives
        }
    )
    margin_values = sorted(
        {
            round(float(record["margin"]), 4)
            for record in positives + negatives
        }
    )
    score_candidates = score_values[:: max(1, len(score_values) // 100)] + [0.0, 1.0]
    margin_candidates = margin_values[:: max(1, len(margin_values) // 100)] + [0.0, 1.0]
    best: dict[str, Any] | None = None
    for score_threshold in sorted(set(score_candidates)):
        for margin_threshold in sorted(set(margin_candidates)):
            false_unknown_accepts = sum(
                record["similarity"] >= score_threshold and record["margin"] >= margin_threshold
                for record in negatives
            )
            accepted_wrong = sum(
                is_wrong(record)
                and record["similarity"] >= score_threshold
                and record["margin"] >= margin_threshold
                for record in positives
            )
            if false_unknown_accepts or accepted_wrong:
                continue
            accepted_correct = sum(
                record[target_field]
                and record["similarity"] >= score_threshold
                and record["margin"] >= margin_threshold
                for record in positives
            )
            candidate = {
                "target": target_field,
                "acceptance_threshold": score_threshold,
                "minimum_margin": margin_threshold,
                "accepted_correct": accepted_correct,
                "positive_count": len(positives),
                "accepted_correct_rate": accepted_correct / len(positives),
                "accepted_wrong": accepted_wrong,
                "synthetic_unknown_false_accept": false_unknown_accepts,
                "unresolved_excluded_count": sum(
                    record["predicted_p_item_id"] == "UNKNOWN" for record in positives
                ),
            }
            if best is None or (
                candidate["accepted_correct"], -score_threshold, -margin_threshold
            ) > (best["accepted_correct"], -best["acceptance_threshold"], -best["minimum_margin"]):
                best = candidate
    return best or {
        "status": "NO_ZERO_ERROR_THRESHOLD",
        "accepted_correct": 0,
        "positive_count": len(positives),
    }


def summarise_calibrated_exact_ids(
    records: list[dict[str, Any]], calibration: dict[str, Any]
) -> dict[str, Any]:
    if "acceptance_threshold" not in calibration or "minimum_margin" not in calibration:
        return {"status": "NO_ZERO_ERROR_THRESHOLD"}
    score_threshold = float(calibration["acceptance_threshold"])
    margin_threshold = float(calibration["minimum_margin"])
    result: dict[str, Any] = {}
    for perturbation in sorted({record["perturbation"] for record in records}):
        subset = [record for record in records if record["perturbation"] == perturbation]
        accepted = [
            record
            for record in subset
            if record["predicted_p_item_id"] != "UNKNOWN"
            and record["similarity"] >= score_threshold
            and record["margin"] >= margin_threshold
        ]
        result[perturbation] = {
            "query_count": len(subset),
            "accepted_count": len(accepted),
            "accepted_rate": len(accepted) / len(subset),
            "accepted_correct_count": sum(record["exact_correct"] for record in accepted),
            "accepted_wrong_count": sum(not record["exact_correct"] for record in accepted),
            "rejected_or_unresolved_count": len(subset) - len(accepted),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate TASK-085 P-item embedding retrieval")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-count", type=int, default=100)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset_path = args.dataset_manifest.resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_root = dataset_path.parent
    reference_root = dataset_root / str(dataset["reference_icons"]["root"])
    rows = json.loads((dataset_root / str(dataset["mapping"]["crosswalk_path"])).read_text(encoding="utf-8"))
    embedder = OnnxPItemEmbedder(
        run_dir / run_manifest["model"]["path"],
        expected_model_sha256=run_manifest["model"]["sha256"],
        source_color_order="RGB",
    )
    gallery = PItemEmbeddingGallery.load(
        run_dir / run_manifest["gallery"]["path"], run_dir / "manifest.json"
    )
    recognizer = PItemEmbeddingRecognizer(
        embedder,
        gallery,
        classes_sha256=run_manifest["gallery"]["class_table_sha256"],
        upgrade_maximum_distance=0.05,
        upgrade_minimum_margin=0.002,
    )

    records = []
    per_perturbation: dict[str, dict[str, Any]] = {}
    for row in rows:
        expected = str(row["business_id"])
        expected_group = str(row["asset_name"])
        with Image.open(reference_root / f"{expected}.png") as source:
            variants = perturbations(source)
        for perturbation, image in variants.items():
            array = np.asarray(image)
            started = time.perf_counter()
            prediction = recognizer.classify(
                array,
                (0, 0, 130, 130),
                frame_id=f"{expected}:{perturbation}",
                plan=str(row["plan"]),
                acceptance_threshold=1e-6,
                minimum_margin=0.0,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            top5_correct = any(expected in candidates for candidates in prediction.top_k_p_item_ids)
            record = {
                "p_item_id": int(expected),
                "perturbation": perturbation,
                "expected_visual_group": expected_group,
                "predicted_visual_group": prediction.visual_group_id,
                "predicted_p_item_id": prediction.predicted_p_item_id,
                "visual_correct": prediction.visual_group_id == expected_group,
                "exact_correct": prediction.predicted_p_item_id == expected,
                "top5_correct": top5_correct,
                "similarity": prediction.confidence,
                "margin": prediction.margin,
                "reason": prediction.reason,
                "latency_ms": latency_ms,
            }
            records.append(record)
    for name in sorted({record["perturbation"] for record in records}):
        subset = [record for record in records if record["perturbation"] == name]
        per_perturbation[name] = {
            "query_count": len(subset),
            "top1_visual_accuracy": sum(record["visual_correct"] for record in subset) / len(subset),
            "top1_exact_id_accuracy": sum(record["exact_correct"] for record in subset) / len(subset),
            "top5_exact_id_recall": sum(record["top5_correct"] for record in subset) / len(subset),
            "unknown_or_unresolved_count": sum(record["predicted_p_item_id"] == "UNKNOWN" for record in subset),
            "wrong_exact_id_count": sum(
                record["predicted_p_item_id"] != "UNKNOWN" and not record["exact_correct"]
                for record in subset
            ),
            "latency_p50_ms": percentile([record["latency_ms"] for record in subset], 50),
            "latency_p95_ms": percentile([record["latency_ms"] for record in subset], 95),
            "failures": [record for record in subset if not record["exact_correct"]],
        }

    negatives = []
    for name, image in negative_images(850418, args.negative_count):
        vector = embedder.embed(np.asarray(image), (0, 0, 130, 130))
        hits = gallery.search(vector, top_k=5)
        negatives.append(
            {
                "name": name,
                "similarity": hits[0].similarity,
                "margin": hits[0].similarity - hits[1].similarity,
                "top_visual_group": hits[0].visual_group_id,
            }
        )
    visual_calibration = calibrate_thresholds(
        [record for record in records if record["perturbation"] != "canonical"],
        negatives,
        target_field="visual_correct",
    )
    exact_calibration = calibrate_thresholds(
        [record for record in records if record["perturbation"] != "canonical"],
        negatives,
        target_field="exact_correct",
    )
    payload = {
        "schema_version": 1,
        "model_sha256": run_manifest["model"]["sha256"],
        "gallery_sha256": run_manifest["gallery"]["sha256"],
        "dataset_manifest_sha256": run_manifest["source"]["dataset_manifest_sha256"],
        "business_id_count": len(rows),
        "visual_identity_count": len(set(row["asset_name"] for row in rows)),
        "per_perturbation": per_perturbation,
        "upgrade_marker_metric_candidates": evaluate_upgrade_marker_metrics(
            rows, reference_root
        ),
        "synthetic_unknown": {
            "count": len(negatives),
            "similarity_max": max(record["similarity"] for record in negatives),
            "margin_max": max(record["margin"] for record in negatives),
            "records": negatives,
        },
        "offline_zero_error_visual_calibration": visual_calibration,
        "offline_zero_error_exact_id_calibration": exact_calibration,
        "calibrated_exact_id_summary": summarise_calibrated_exact_ids(
            records, exact_calibration
        ),
        "synthetic_upgrade_operating_point": {
            "maximum_distance": 0.05,
            "minimum_margin": 0.002,
            "status": "EVALUATION_ONLY_PENDING_FROZEN_JJC",
        },
        "limitations": [
            "synthetic negatives do not replace a frozen TASK-400 JJC screenshot set",
            "runtime acceptance thresholds remain unapproved until the JJC freeze is evaluated",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "canonical_exact": per_perturbation["canonical"]["top1_exact_id_accuracy"],
                "low16_top1": per_perturbation["low_resolution_16"]["top1_exact_id_accuracy"],
                "low16_top5": per_perturbation["low_resolution_16"]["top5_exact_id_recall"],
                "visual_calibration": visual_calibration,
                "exact_id_calibration": exact_calibration,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
