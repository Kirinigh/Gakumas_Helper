"""Persist the frozen arena-only reference experiment; no training or game input."""
from __future__ import annotations

import sys
import json
import shutil
import argparse
from pathlib import Path

import numpy as np
from diagnose_arena_candidate_recall import sha, read, read_bgr, load_reference_paths

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))
from card_selection.embedding import EmbeddingCardRecognizer


def extend_arrays(arrays, frozen_arrays, rendered):
    """Append only frozen reference membership, preserving current raw rows exactly."""
    count = len(frozen_arrays["class_names"])
    for key, old in frozen_arrays.items():
        if not np.array_equal(arrays[key][:count], old):
            raise ValueError(f"frozen baseline prefix changed: {key}")
    if rendered.shape != (count, 128) or not np.isfinite(rendered).all():
        raise ValueError("invalid full UI vectors")
    if any(str(n).endswith("_full_ui") for n in arrays["class_names"]):
        raise ValueError("full UI extension already applied")
    result = {}
    for key, value in arrays.items():
        extra = rendered if key == "embeddings" else frozen_arrays[key]
        if key == "class_names":
            extra = np.asarray([str(n) + "_full_ui" for n in extra])
        result[key] = np.concatenate((value, extra))
    return result


def export(current, frozen, report_path, output):
    if output.exists():
        raise FileExistsError(output)
    report = read(report_path)["components"]["embedding/arena_card"]
    old = read(frozen / "manifest.json")
    manifest = read(current / "manifest.json")
    if manifest["component"] != "arena_custom_card_embedding":
        raise ValueError("arena gallery only")
    for key in ("model", "gallery"):
        if sha(frozen / old[key]["path"]) != report[key + "_sha256"]:
            raise ValueError(f"frozen {key} binding changed")
    if manifest["model"]["sha256"] != report["model_sha256"]:
        raise ValueError("encoder changed")
    recognizer = EmbeddingCardRecognizer.load(current)
    with np.load(current / manifest["gallery"]["path"], allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    with np.load(frozen / old["gallery"]["path"], allow_pickle=False) as data:
        frozen_arrays = {k: data[k] for k in data.files}
    sources = []
    for receipt in report["frozen_reference_sources"]:
        path = Path(receipt["path"])
        path = path if path.is_absolute() else ROOT / path
        if sha(path) != receipt["sha256"]:
            raise ValueError("frozen source manifest changed")
        sources.append(path)
    paths, receipts = load_reference_paths(sources, frozen_arrays["class_names"])
    vectors = []
    for index, path in enumerate(paths):
        image = read_bgr(path)
        vectors.append(recognizer.embedder.embed(image, (0, 0, image.shape[1], image.shape[0])))
        if index % 500 == 0:
            print(f"encoded {index}/{len(paths)}", flush=True)
    extended = extend_arrays(arrays, frozen_arrays, np.stack(vectors))
    output.mkdir(parents=True)
    preserved = output / "preserved_base"
    preserved.mkdir()
    shutil.copyfile(current / "manifest.json", preserved / "manifest.json")
    shutil.copyfile(current / manifest["gallery"]["path"], preserved / manifest["gallery"]["path"])
    shutil.copyfile(current / manifest["model"]["path"], output / manifest["model"]["path"])
    gallery = output / manifest["gallery"]["path"]
    np.savez_compressed(gallery, **extended)
    classes = output / manifest["gallery"]["class_table_path"]
    classes.write_text(json.dumps(extended["class_names"].tolist(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    baseline_manifest_sha = sha(current / "manifest.json")
    manifest.pop("promotion", None)
    manifest["validation"] = {
        "state": "USER_AUTHORIZED_ENABLE_VALIDATION_PENDING",
        "pending": ["real_unknown_truncated_wrong_slot_rejection", "own_nine_full_read_p50_p95_102_percent",
                    "opponent_full_flow", "current_297_and_553_failure_pairs"],
        "frozen_diagnostic_report_sha256": sha(report_path),
    }
    manifest["production_handoff"] = {
        "status": "READY_WITH_REAL_SAMPLES_PENDING",
        "reason": "2026-09-21 user explicitly authorized enablement before outstanding follow-up validation; not an acceptance pass",
    }
    manifest["build"] = {
        "base_dataset_manifest_sha256": manifest["build"]["base_dataset_manifest_sha256"],
        "base_dataset_lineage_manifest_sha256s": manifest["build"]["base_dataset_lineage_manifest_sha256s"],
        "tool_contract": "task095-arena-full-ui-union-v1",
        "mode": "GALLERY_ONLY", "base_manifest_sha256": baseline_manifest_sha,
        "base_gallery_sha256": manifest["gallery"]["sha256"],
        "preserved_rows": len(arrays["class_names"]), "appended_rows": len(paths),
        "frozen_gallery_sha256": report["gallery_sha256"],
        "reference_manifest_sha256s": [r["sha256"] for r in receipts],
        "policy": "frozen full UI reference membership; base model unchanged; no per-ID exceptions",
        "upgrade_marker_policy": "duplicate original markers; arena candidate callers disable upgrade resolution",
    }
    manifest["gallery"].update(sha256=sha(gallery), bytes=gallery.stat().st_size,
                               class_table_sha256=sha(classes), class_table_bytes=classes.stat().st_size)
    manifest["source"]["image_class_count"] = len(extended["class_names"])
    manifest["source"]["source_domain"] = "OFFICIAL_RAW_CARD_ART_WITH_FROZEN_FULL_UI_REFERENCES"
    manifest["source"]["training_target"] = "encoder unchanged; gallery reference augmentation only"
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    loaded = EmbeddingCardRecognizer.load(output)
    assert len(loaded.gallery.card_ids) == len(extended["class_names"])
    print(json.dumps(manifest["build"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("current", "frozen", "report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    export(args.current, args.frozen, args.report, args.output)
