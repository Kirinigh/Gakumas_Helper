"""Build an offline candidate from a frozen extension's complete UI icons.

Preserve every accepted row and the encoder. This tool does not promote assets;
source-image self queries are not independent recognition acceptance evidence.
"""
from __future__ import annotations

import sys
import json
import shutil
import hashlib
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))
from card_selection.embedding import EmbeddingCardRecognizer
from diagnose_arena_candidate_recall import read_bgr


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def append_rows(arrays, names, vectors):
    if not names or len(set(names)) != len(names):
        raise ValueError("empty or duplicate source names")
    current = arrays["class_names"].tolist()
    indices = []
    for name in names:
        if current.count(name) != 1 or name + "_full_ui" in current:
            raise ValueError("missing, nonunique or already appended source")
        indices.append(current.index(name))
    if vectors.shape != (len(names), 128) or not np.isfinite(vectors).all():
        raise ValueError("invalid vectors")
    result = {}
    for key, value in arrays.items():
        extra = value[indices]
        if key == "embeddings":
            extra = vectors
        elif key == "class_names":
            extra = np.asarray([n + "_full_ui" for n in names])
        result[key] = np.concatenate((value, extra))
    return result


def build(current, dataset, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((current / "manifest.json").read_bytes())
    source = json.loads(dataset.read_bytes())
    icons = source["icons"]
    if icons["count"] != len(icons["images"]):
        raise ValueError("source count mismatch")
    if sorted(r["business_id"] for r in icons["images"]) != sorted(source["scope"]["business_ids"]):
        raise ValueError("source scope mismatch")
    recognizer = EmbeddingCardRecognizer.load(current)
    with np.load(current / manifest["gallery"]["path"], allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    names, vectors, receipts = [], [], []
    for row in icons["images"]:
        path = (dataset.parent / icons["root"] / row["path"]).resolve()
        if not path.is_relative_to(dataset.parent.resolve()) or sha(path) != row["sha256"]:
            raise ValueError("source path or digest mismatch")
        name = row["class_name"]
        index = arrays["class_names"].tolist().index(name)
        if str(arrays["card_ids"][index]) != str(row["business_id"]):
            raise ValueError("source identity mismatch")
        image = read_bgr(path)
        if image.shape[:2] != (row["height"], row["width"]):
            raise ValueError("source dimensions mismatch")
        vectors.append(recognizer.embedder.embed(image, (0, 0, image.shape[1], image.shape[0])))
        names.append(name)
        receipts.append({"class_name": name, "sha256": sha(path)})
    extended = append_rows(arrays, names, np.stack(vectors))
    output.mkdir(parents=True)
    shutil.copyfile(current / manifest["model"]["path"], output / manifest["model"]["path"])
    gallery = output / manifest["gallery"]["path"]
    np.savez_compressed(gallery, **extended)
    classes = output / manifest["gallery"]["class_table_path"]
    classes.write_text(json.dumps(extended["class_names"].tolist(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest.pop("promotion", None)
    manifest["validation"] = {"state": "OFFLINE_CANDIDATE_PENDING", "pending": [
        "independent_failure_replay", "normal_upgrade_resolution", "unknown_rejection",
        "full_read_p50_p95_within_102_percent"]}
    manifest["production_handoff"] = {"status": "PRODUCTION_HANDOFF_BLOCKED", "reason": "independent validation pending"}
    manifest["build"] = {"tool_contract": "task095-verified-ui-append-v1", "mode": "GALLERY_ONLY",
                         "base_manifest_sha256": sha(current / "manifest.json"),
                         "base_gallery_sha256": manifest["gallery"]["sha256"],
                         "dataset_manifest_sha256": sha(dataset), "references": receipts,
                         "preserved_rows": len(arrays["class_names"]), "appended_rows": len(names)}
    manifest["gallery"].update(sha256=sha(gallery), bytes=gallery.stat().st_size,
                               class_table_sha256=sha(classes), class_table_bytes=classes.stat().st_size)
    manifest["source"]["image_class_count"] = len(extended["class_names"])
    manifest["source"]["source_domain"] = "OFFICIAL_RAW_ART_WITH_VERIFIED_UI_REFERENCES"
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    EmbeddingCardRecognizer.load(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("current", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.current, args.dataset, args.output)
