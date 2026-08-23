from __future__ import annotations

import sys
import json
import time
import argparse
import statistics
from typing import Any
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.card_selection.model import sha256_file
from agent.card_selection.embedding import (
    EmbeddingGallery,
    OnnxCardEmbedder,
    resolve_card_identity,
)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a TASK-075 ONNX encoder and gallery")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-color-order", choices=("RGB", "BGR"), default="RGB")
    parser.add_argument("--acceptance-threshold", type=float)
    parser.add_argument("--minimum-margin", type=float)
    args = parser.parse_args()
    if (args.acceptance_threshold is None) != (args.minimum_margin is None):
        raise ValueError("acceptance threshold and minimum margin must be supplied together")
    thresholds_supplied = args.acceptance_threshold is not None

    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    gallery = EmbeddingGallery.load(args.run_dir / manifest["gallery"]["path"], manifest_path)
    embedder = OnnxCardEmbedder(
        args.run_dir / manifest["model"]["path"],
        expected_model_sha256=manifest["model"]["sha256"],
        source_color_order=args.source_color_order,
    )
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    if dataset.get("schema_version") != 1 or not dataset.get("samples"):
        raise ValueError("dataset must use schema_version 1 and contain samples")

    image_cache: dict[Path, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for sample in dataset["samples"]:
        image_path = Path(sample["image_path"])
        if not image_path.is_absolute():
            image_path = (args.dataset.parent / image_path).resolve()
        if image_path not in image_cache:
            with Image.open(image_path) as source:
                image_cache[image_path] = np.asarray(source.convert("RGB"))
        started = time.perf_counter()
        hits = gallery.search(embedder.embed(image_cache[image_path], tuple(sample["box"])), top_k=5)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        similarity = hits[0].similarity
        margin = similarity - hits[1].similarity
        top5_card_ids = [list(hit.candidate_card_ids or (hit.card_id,)) for hit in hits]
        expected_unknown = bool(sample.get("expected_unknown", False))
        expected = None if expected_unknown else str(sample["expected_card_id"])
        resolved_id: str | None = None
        predicted_upgraded: bool | None = None
        resolution_error: str | None = None
        if not expected_unknown:
            try:
                predicted_upgraded = gallery.infer_upgrade_state(
                    image_cache[image_path],
                    tuple(sample["box"]),
                    hits[0],
                    source_color_order=args.source_color_order,
                )
                resolved_id = resolve_card_identity(
                    hits,
                    plan=str(sample.get("plan", "free")),
                    upgraded=predicted_upgraded,
                ).card_id
            except ValueError as error:
                resolution_error = str(error)
        accepted = (
            thresholds_supplied
            and resolution_error is None
            and similarity >= args.acceptance_threshold
            and margin >= args.minimum_margin
        )
        row = {
            "sample_id": str(sample["sample_id"]),
            "expected_unknown": expected_unknown,
            "expected_card_id": expected,
            "retrieved_card_id": resolved_id,
            "predicted_card_id": resolved_id if accepted else "UNKNOWN" if thresholds_supplied else resolved_id,
            "retrieval_correct": expected is not None and resolved_id == expected,
            "accepted": accepted if thresholds_supplied else None,
            "resolution_error": resolution_error,
            "expected_upgraded": sample.get("upgraded"),
            "predicted_upgraded": predicted_upgraded,
            "upgrade_state_correct": (
                predicted_upgraded == sample.get("upgraded") if not expected_unknown else None
            ),
            "top5_card_ids": top5_card_ids,
            "top5_similarities": [hit.similarity for hit in hits],
            "similarity": similarity,
            "margin": margin,
            "latency_ms": elapsed_ms,
        }
        rows.append(row)

    positive_rows = [row for row in rows if not row["expected_unknown"]]
    negative_rows = [row for row in rows if row["expected_unknown"]]
    wrong = [row for row in positive_rows if not row["retrieval_correct"]]
    known_unknown = [row for row in positive_rows if thresholds_supplied and not row["accepted"]]
    false_accepts = [row for row in negative_rows if thresholds_supplied and row["accepted"]]
    latencies = [row["latency_ms"] for row in rows]
    steady_latencies = latencies[1:] or latencies
    top5_correct_count = sum(
        any(row["expected_card_id"] in group for group in row["top5_card_ids"])
        for row in positive_rows
    )
    calibration = (
        "PASS"
        if thresholds_supplied
        and positive_rows
        and negative_rows
        and not wrong
        and not known_unknown
        and not false_accepts
        else "FAIL"
        if thresholds_supplied and positive_rows and negative_rows
        else "PENDING"
    )
    payload = {
        "schema_version": 2,
        "dataset_id": dataset.get("dataset_id"),
        "model_sha256": embedder.model_sha256,
        "gallery_sha256": gallery.gallery_sha256,
        "sample_count": len(rows),
        "known_card_count": len(positive_rows),
        "unknown_sample_count": len(negative_rows),
        "top1_correct_count": len(positive_rows) - len(wrong),
        "top1_accuracy": (
            (len(positive_rows) - len(wrong)) / len(positive_rows) if positive_rows else None
        ),
        "top5_correct_count": top5_correct_count,
        "top5_accuracy": top5_correct_count / len(positive_rows) if positive_rows else None,
        "wrong_id_count": len(wrong),
        "known_card_unknown_count": len(known_unknown) if thresholds_supplied else None,
        "unknown_false_accept_count": len(false_accepts) if thresholds_supplied else None,
        "acceptance_calibration": calibration,
        "thresholds": {
            "acceptance": args.acceptance_threshold,
            "minimum_margin": args.minimum_margin,
        },
        "latency_ms": {
            "cold_first": latencies[0] if latencies else None,
            "steady_median": statistics.median(steady_latencies) if steady_latencies else None,
            "steady_p95": percentile(steady_latencies, 95),
            "steady_max": max(steady_latencies) if steady_latencies else None,
            "note": "cold_first includes first-session initialisation; steady values exclude it",
        },
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["validation"].update(
        {
            "independent_dmm_atlas_gate": "PASS" if positive_rows and not wrong else "FAIL",
            "acceptance_calibration": calibration,
            "independent_dmm_atlas_path": args.output.name,
            "independent_dmm_atlas_sha256": sha256_file(args.output),
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}, ensure_ascii=False))
    return 0 if not wrong else 1


if __name__ == "__main__":
    raise SystemExit(main())
