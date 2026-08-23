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

from agent.live_decision import (
    LiveCardLayoutError,
    build_live_card_views,
    normalise_live_candidates,
    preprocess_live_card_view,
)
from agent.card_selection import CandidateBox, EmbeddingCardRecognizer


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"))


def _candidate(raw: dict[str, Any], index: int) -> CandidateBox:
    return CandidateBox(
        slot=index,
        box=tuple(int(value) for value in raw["box"]),
        detector_label=str(raw.get("label", "cards")),
        detector_score=float(raw.get("score", 1.0)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate partial Live card identity")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptance-threshold", type=float)
    parser.add_argument("--minimum-margin", type=float)
    args = parser.parse_args()
    if (args.acceptance_threshold is None) != (args.minimum_margin is None):
        raise ValueError("acceptance threshold and minimum margin must be supplied together")

    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    if dataset.get("schema_version") != 1 or not dataset.get("samples"):
        raise ValueError("dataset must use schema_version 1 and contain samples")
    recognizer = EmbeddingCardRecognizer.load(args.model_root, source_color_order="RGB")
    rows: list[dict[str, Any]] = []
    known_count = 0
    negative_count = 0
    known_unknown_count = 0
    false_accept_count = 0
    latencies: list[float] = []

    for sample in dataset["samples"]:
        image_path = Path(sample["image_path"])
        if not image_path.is_absolute():
            image_path = (args.dataset.parent / image_path).resolve()
        image = _load_image(image_path)
        candidates = normalise_live_candidates(
            _candidate(raw, index) for index, raw in enumerate(sample["candidates"])
        )
        expected_groups = tuple(
            tuple(str(card_id) for card_id in group)
            for group in sample.get("expected_visual_groups", ())
        )
        expected_unknown = bool(sample.get("expected_unknown", False))
        if expected_unknown:
            negative_count += 1
        else:
            known_count += len(expected_groups)
        started = time.perf_counter()
        try:
            views = build_live_card_views(image.shape, candidates)
        except LiveCardLayoutError as error:
            if not expected_unknown:
                known_unknown_count += len(expected_groups)
            rows.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "expected_unknown": expected_unknown,
                    "accepted": False,
                    "reason": str(error),
                    "cards": [],
                }
            )
            continue

        cards = []
        sample_accepted = False
        for view in views:
            vector = recognizer.embedder.embed_tensor(
                preprocess_live_card_view(image, view, source_color_order="RGB")
            )
            hits = recognizer.gallery.search(vector, top_k=5)
            similarity = hits[0].similarity
            margin = similarity - hits[1].similarity
            accepted = (
                args.acceptance_threshold is not None
                and similarity >= args.acceptance_threshold
                and margin >= args.minimum_margin
            )
            sample_accepted = sample_accepted or accepted
            expected = expected_groups[view.slot] if view.slot < len(expected_groups) else ()
            top_groups = tuple(hit.candidate_card_ids or (hit.card_id,) for hit in hits)
            top1_correct = bool(expected) and bool(set(expected) & set(top_groups[0]))
            top5_correct = bool(expected) and any(set(expected) & set(group) for group in top_groups)
            if expected and not accepted and args.acceptance_threshold is not None:
                known_unknown_count += 1
            cards.append(
                {
                    "slot": view.slot,
                    "expected_visual_group": list(expected),
                    "top1_visual_group": list(top_groups[0]),
                    "top5_visual_groups": [list(group) for group in top_groups],
                    "top1_correct": top1_correct,
                    "top5_correct": top5_correct,
                    "similarity": similarity,
                    "margin": margin,
                    "accepted": accepted if args.acceptance_threshold is not None else None,
                    "visible_ratio": view.visible_ratio,
                    "view_mode": view.view_mode,
                    "visible_box": list(view.visible_box),
                    "canvas_size": list(view.canvas_size),
                }
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed_ms)
        if expected_unknown and sample_accepted:
            false_accept_count += 1
        rows.append(
            {
                "sample_id": str(sample["sample_id"]),
                "expected_unknown": expected_unknown,
                "accepted": sample_accepted if args.acceptance_threshold is not None else None,
                "reason": "evaluated",
                "latency_ms": elapsed_ms,
                "cards": cards,
            }
        )

    known_cards = [card for row in rows if not row["expected_unknown"] for card in row["cards"]]
    top1_correct = sum(card["top1_correct"] for card in known_cards)
    top5_correct = sum(card["top5_correct"] for card in known_cards)
    thresholds_supplied = args.acceptance_threshold is not None
    calibration_state = (
        "PASS"
        if thresholds_supplied
        and known_count > 0
        and negative_count > 0
        and top1_correct == known_count
        and known_unknown_count == 0
        and false_accept_count == 0
        else "FAIL"
        if thresholds_supplied and known_count > 0 and negative_count > 0
        else "PENDING"
    )
    payload = {
        "schema_version": 1,
        "dataset_id": dataset.get("dataset_id"),
        "model_sha256": recognizer.model_sha256,
        "gallery_sha256": recognizer.gallery_sha256,
        "known_card_count": known_count,
        "negative_sample_count": negative_count,
        "top1_correct_count": top1_correct,
        "top1_accuracy": top1_correct / known_count if known_count else None,
        "top5_correct_count": top5_correct,
        "top5_accuracy": top5_correct / known_count if known_count else None,
        "known_card_unknown_count": known_unknown_count if thresholds_supplied else None,
        "unknown_false_accept_count": false_accept_count if thresholds_supplied else None,
        "acceptance_calibration": calibration_state,
        "latency_ms": {
            "median": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 95),
            "maximum": max(latencies) if latencies else None,
        },
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}, ensure_ascii=False))
    return 0 if top1_correct == known_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
