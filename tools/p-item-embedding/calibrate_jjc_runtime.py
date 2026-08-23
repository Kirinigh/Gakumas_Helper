from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections.abc import Mapping, Iterable, Sequence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate low-dimensional JJC P-item evidence and reject unsafe thresholds",
    )
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _prediction(item: Mapping[str, object]) -> Mapping[str, object]:
    nested = item.get("prediction")
    return nested if isinstance(nested, Mapping) else item


def _completeness(item: Mapping[str, object]) -> Mapping[str, object]:
    nested = item.get("completeness")
    return nested if isinstance(nested, Mapping) else {}


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _top_k_ids(prediction: Mapping[str, object]) -> tuple[int, ...]:
    raw = prediction.get("top_k_p_item_ids")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    result: list[int] = []
    for group in raw:
        values = group if isinstance(group, Sequence) and not isinstance(group, (str, bytes)) else (group,)
        for value in values:
            text = str(value)
            if text.isdigit():
                result.append(int(text))
    return tuple(dict.fromkeys(result))


def _reference_truth(slot: Mapping[str, object]) -> tuple[int | None, float | None, float | None]:
    repeated = slot.get("repeated_predictions")
    if not isinstance(repeated, Sequence) or not repeated:
        return None, None, None
    first = repeated[0]
    if not isinstance(first, Mapping):
        return None, None, None
    reference = first.get("reference_match")
    if not isinstance(reference, Mapping) or reference.get("status") != "MEASURED":
        return None, None, None
    top_k = reference.get("top_k")
    if not isinstance(top_k, Sequence) or not top_k or not isinstance(top_k[0], Mapping):
        return None, None, None
    raw_id = top_k[0].get("p_item_id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
        return None, None, None
    return raw_id, _float(top_k[0].get("similarity")), _float(reference.get("top1_gap"))


def _range(values: Iterable[float]) -> dict[str, float | None]:
    ordered = sorted(values)
    return {
        "minimum": None if not ordered else round(ordered[0], 6),
        "maximum": None if not ordered else round(ordered[-1], 6),
    }


def aggregate_reports(reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    positives: list[dict[str, object]] = []
    negatives: list[dict[str, object]] = []
    seen_positive: set[tuple[str, int, int, int]] = set()
    seen_negative: set[tuple[str, int, int, str, int]] = set()
    for report_index, report in enumerate(reports):
        report_id = str(report.get("capture_id") or report.get("run_id") or f"report-{report_index}")
        steps = report.get("steps")
        if not isinstance(steps, Sequence):
            continue
        for step in steps:
            if not isinstance(step, Mapping) or not str(step.get("name", "")).endswith("p_item_calibration"):
                continue
            members = step.get("members")
            if not isinstance(members, Sequence):
                continue
            for member in members:
                if not isinstance(member, Mapping):
                    continue
                stage = int(member.get("stage_number", 0))
                member_slot = int(member.get("member_slot", 0))
                slots = member.get("p_item_predictions")
                if isinstance(slots, Sequence):
                    for slot in slots:
                        if not isinstance(slot, Mapping):
                            continue
                        slot_index = int(slot.get("slot", 0))
                        key = (report_id, stage, member_slot, slot_index)
                        if key in seen_positive:
                            continue
                        seen_positive.add(key)
                        repeated = slot.get("repeated_predictions")
                        if not isinstance(repeated, Sequence) or not repeated:
                            continue
                        rows = [item for item in repeated if isinstance(item, Mapping)]
                        completeness = [_completeness(item) for item in rows]
                        foreground = [
                            value
                            for item in completeness
                            if (value := _float(item.get("foreground_fraction"))) is not None
                        ]
                        opposite = [
                            value
                            for item in completeness
                            if (
                                value := _float(item.get("minimum_opposite_half_fraction"))
                            )
                            is not None
                        ]
                        truth, reference_similarity, reference_gap = _reference_truth(slot)
                        confirmed_empty = bool(foreground) and max(foreground) <= 0.05
                        if confirmed_empty:
                            truth = 0
                        prediction = _prediction(rows[0])
                        raw_id = prediction.get("p_item_id")
                        predicted = int(raw_id) if str(raw_id).isdigit() else None
                        positives.append(
                            {
                                "truth": truth,
                                "predicted": predicted,
                                "top_k": _top_k_ids(prediction),
                                "confidence": _float(prediction.get("confidence")),
                                "margin": _float(prediction.get("margin")),
                                "foreground_minimum": min(foreground) if foreground else None,
                                "opposite_half_minimum": min(opposite) if opposite else None,
                                "reference_similarity": reference_similarity,
                                "reference_gap": reference_gap,
                            }
                        )
                probes = member.get("p_item_negative_probes")
                if not isinstance(probes, Sequence):
                    continue
                for probe_index, probe in enumerate(probes):
                    if not isinstance(probe, Mapping):
                        continue
                    kind = str(probe.get("kind", "unknown"))
                    key = (report_id, stage, member_slot, kind, probe_index)
                    if key in seen_negative:
                        continue
                    seen_negative.add(key)
                    observations = probe.get("repeated_observations")
                    rows = (
                        [item for item in observations if isinstance(item, Mapping)]
                        if isinstance(observations, Sequence)
                        else [probe]
                    )
                    predictions = [_prediction(item) for item in rows]
                    completeness = [_completeness(item) for item in rows]
                    confidence = [
                        value
                        for item in predictions
                        if (value := _float(item.get("confidence"))) is not None
                    ]
                    margins = [
                        value
                        for item in predictions
                        if (value := _float(item.get("margin"))) is not None
                    ]
                    foreground = [
                        value
                        for item in completeness
                        if (value := _float(item.get("foreground_fraction"))) is not None
                    ]
                    opposite = [
                        value
                        for item in completeness
                        if (
                            value := _float(item.get("minimum_opposite_half_fraction"))
                        )
                        is not None
                    ]
                    negatives.append(
                        {
                            "kind": kind,
                            "confidence": max(confidence) if confidence else None,
                            "margin": max(margins) if margins else None,
                            "foreground": max(foreground) if foreground else None,
                            "opposite_half": max(opposite) if opposite else None,
                        }
                    )

    nonempty = [item for item in positives if item["truth"] not in {None, 0}]
    empty = [item for item in positives if item["truth"] == 0]
    top1_correct = sum(item["predicted"] == item["truth"] for item in nonempty)
    top5_correct = sum(item["truth"] in item["top_k"] for item in nonempty)
    wrong = [item for item in nonempty if item["predicted"] != item["truth"]]
    negative_confidence = [item["confidence"] for item in negatives if item["confidence"] is not None]
    positive_confidence = [item["confidence"] for item in nonempty if item["confidence"] is not None]
    negative_margin = [item["margin"] for item in negatives if item["margin"] is not None]
    positive_margin = [item["margin"] for item in nonempty if item["margin"] is not None]
    positive_opposite = [
        item["opposite_half_minimum"]
        for item in nonempty
        if item["opposite_half_minimum"] is not None
    ]
    negative_opposite = [
        item["opposite_half"]
        for item in negatives
        if item["opposite_half"] is not None
    ]
    confidence_overlaps = bool(
        positive_confidence
        and negative_confidence
        and min(positive_confidence) <= max(negative_confidence)
    )
    margin_overlaps = bool(
        positive_margin
        and negative_margin
        and min(positive_margin) <= max(negative_margin)
    )
    embedding_approved = not wrong and not confidence_overlaps and not margin_overlaps
    return {
        "schema_version": 1,
        "independent_slots": len(positives),
        "nonempty_slots": len(nonempty),
        "confirmed_empty_slots": len(empty),
        "negative_observations": len(negatives),
        "embedding": {
            "top1_correct": top1_correct,
            "top1_total": len(nonempty),
            "top5_contains_truth": top5_correct,
            "top5_total": len(nonempty),
            "stable_wrong_identity_count": len(wrong),
            "positive_confidence": _range(positive_confidence),
            "negative_confidence": _range(negative_confidence),
            "positive_margin": _range(positive_margin),
            "negative_margin": _range(negative_margin),
            "confidence_overlaps": confidence_overlaps,
            "margin_overlaps": margin_overlaps,
            "runtime_approved": embedding_approved,
        },
        "completeness": {
            "positive_opposite_half": _range(positive_opposite),
            "negative_opposite_half": _range(negative_opposite),
            "current_sample_separated": bool(
                positive_opposite
                and negative_opposite
                and min(positive_opposite) > max(negative_opposite)
            ),
        },
        "reference": {
            "measured_nonempty_slots": sum(
                item["reference_similarity"] is not None for item in nonempty
            ),
            "similarity": _range(
                item["reference_similarity"]
                for item in nonempty
                if item["reference_similarity"] is not None
            ),
            "margin": _range(
                item["reference_gap"]
                for item in nonempty
                if item["reference_gap"] is not None
            ),
        },
        "decision": {
            "embedding_role": (
                "authoritative" if embedding_approved else "diagnostic_only"
            ),
            "identity_authority": "full_rendered_reference",
            "incomplete_or_ambiguous_action": "bounded_detail_or_fail_closed",
            "upgraded_marker_runtime_approved": False,
            "reason": (
                "embedding distributions separate without observed errors"
                if embedding_approved
                else "stable identity errors or positive/negative threshold overlap"
            ),
        },
    }


def main() -> int:
    args = _parse_args()
    reports = [
        json.loads(path.read_text(encoding="utf-8-sig")) for path in args.report
    ]
    result = aggregate_reports(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
