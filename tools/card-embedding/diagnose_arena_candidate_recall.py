"""Read-only replay of saved arena transactions; no Maa, capture, OCR or training.

Build a uniform full-UI reference experiment in memory from frozen datasets.
Never promote it: saved transactions are diagnostic evidence, not release gates.
"""

from __future__ import annotations

import re
import ast
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path

import cv2
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def read_bgr(path):
    """Decode via Python paths so Windows attachment names may contain Unicode."""
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)


def interaction_function(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MaaArenaReaderBackend")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_skill_card_interaction_box")
    node.decorator_list = []
    scope = {"ArenaReaderError": ValueError}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<installed-interaction-geometry>", "exec"), scope)
    return scope[node.name]


def recover_box(metadata, shape, interaction):
    """Invert the logged deterministic body press using installed canonical size.

    This reconstructs the transaction's selected rectangle, not its missing
    original three-frame identity generation or exact historical embedding input.
    """
    height, width = shape[:2]
    w, h = round(width * 0.114), round(height * 0.066)
    actions = [a for a in metadata["actions"] if a["kind"] == "short_press" and a["succeeded"]]
    if not actions:
        raise ValueError("no successful source-card press")
    px, py, pw, ph = actions[0]["box"]
    if (pw, ph) != (1, 1):
        raise ValueError("not a deterministic body press")
    box = (px - round(w * 0.48), py - round(h * 0.48), w, h)
    slot = metadata["card_slot"] - 1
    lefts = (round(width * 0.039), round(width * 0.214)) if metadata["group_index"] == 1 else (round(width * 0.039),)
    if box[0] not in {left + round(width * 0.1264) * slot for left in lefts}:
        raise ValueError("press does not agree with canonical column")
    if tuple(interaction(box)) != (px, py, 1, 1):
        raise ValueError("installed interaction geometry differs")
    logged_box = re.search(r"(?<![a-z_])card_box=(\([^)]*\))", metadata.get("error", ""))
    if logged_box and tuple(ast.literal_eval(logged_box.group(1))) != box:
        raise ValueError("logged card rectangle disagrees with recorded press")
    x, y, w, h = box
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("recovered rectangle outside frame")
    return box


def freeze_cases(evidence_root, cards, interaction, review=None):
    by_name = {}
    for card in cards:
        by_name.setdefault(card["name"], []).append(card["id"])
    cases, excluded = [], []
    for path in sorted(Path(evidence_root).glob("*/evidence.json")):
        meta = read(path)
        decision = (review or {}).get(path.parent.name, {})
        if decision.get("association_confirmed") is not True:
            excluded.append({"transaction": path.parent.name, "reason": "source/detail association not reviewed; not a recall-positive label"})
            continue
        ids = by_name.get(meta.get("card_name"), [])
        confirmed = any(f.get("role") == "title_confirmed_open" for f in meta["frames"])
        # Legacy title evidence must be explicitly reviewed in a private input.
        legacy_752 = decision.get("legacy_full_title_confirmed") is True
        if len(ids) != 1 or not (confirmed or legacy_752):
            excluded.append({"transaction": path.parent.name, "reason": "no unique confirmed title evidence"})
            continue
        image_path = path.parent / "00.png"
        image = read_bgr(image_path)
        if image is None:
            raise ValueError(f"missing frame: {image_path}")
        try:
            box = recover_box(meta, image.shape, interaction)
        except ValueError as error:
            excluded.append({"transaction": path.parent.name, "reason": str(error)})
            continue
        x, y, w, h = box
        cases.append(
            {
                "transaction": path.parent.name,
                "recorded_source_folder": meta.get("folder"),
                "image_path": str(image_path),
                "frame_sha256": sha(image_path),
                "metadata_sha256": sha(path),
                "crop_sha256": hashlib.sha256(image[y : y + h, x : x + w].tobytes()).hexdigest().upper(),
                "card_id": ids[0],
                "name": meta["card_name"],
                "box": box,
                "slot_index": meta["card_slot"] - 1,
                "plan": {1: "sense", 2: "logic", 3: "anomaly"}[meta["stage_number"]],
                "partition": "development" if ids[0] in {99, 752} else "heldout_other_identity",
                "truth_basis": "legacy_full_title_error" if legacy_752 else "recorded_title_confirmed_open",
            }
        )
    return cases, excluded


def load_reference_paths(manifests, names):
    paths, receipts = {}, []
    for path in manifests:
        manifest = read(path)
        receipts.append({"path": str(path), "sha256": sha(path)})
        for row in manifest["icons"]["images"]:
            name = row["class_name"]
            if name in paths:
                raise ValueError("duplicate frozen reference class")
            image = Path(path).parent / manifest["icons"]["root"] / row["path"]
            if sha(image) != row["sha256"]:
                raise ValueError(f"reference digest mismatch: {name}")
            paths[name] = image
    if set(paths) != set(names):
        raise ValueError("full-UI and installed gallery classes differ")
    return [paths[n] for n in names], receipts


def freeze_fresh_cases(manifest_path, cards, interaction):
    """Verify linked source/detail evidence before candidate scoring."""
    manifest = read(manifest_path)
    names = {card["id"]: card["name"] for card in cards}
    cases, scenes, seen = [], set(), set()
    for item in manifest["transactions"]:
        directory = Path(item["directory"])
        evidence_path = directory / "evidence.json"
        if sha(evidence_path).lower() != item["evidence_sha256"].lower():
            raise ValueError("fresh evidence digest mismatch")
        meta = read(evidence_path)
        position = meta["position"]
        identity = item["card_id"]
        scene = f"own-stage{item['stage']}-member{item['member']}"
        expected_position = (item["stage"], item["member"], item["row"], item["slot"])
        actual_position = tuple(position[k] for k in ("stage_number", "member_slot", "group_index", "card_slot"))
        if (actual_position != expected_position or position["team_id"] != "self"
                or item["scene_group"] != scene or item["plan"] != {1: "sense", 2: "logic", 3: "anomaly"}[item["stage"]]):
            raise ValueError("fresh scene/position mismatch")
        if (meta["open_title_id"] != identity or meta["expected_from_cache"] != identity
                or names[identity] != item["name"] or position["card_name"] != names[identity]
                or names[identity] not in meta["detail_text"].splitlines()):
            raise ValueError("fresh source/detail identity mismatch")
        if meta["source_version"] != manifest["version"] or not meta["captured_at"].startswith(manifest["capture_date"]):
            raise ValueError("fresh capture provenance mismatch")
        if meta["outcome"] or any(not a["succeeded"] for a in meta["actions"]):
            raise ValueError("fresh transaction contains failure or unreviewed outcome")
        frames = meta["frames"]
        source = [f for f in frames if f["role"] == "source"]
        opened = [f for f in frames if f["role"] == "opened"]
        if len(source) != 1 or len(opened) != 1:
            raise ValueError("fresh source/opened frame missing or ambiguous")
        for frame in frames:
            if sha(directory / frame["file"]).lower() != frame["sha256"].lower():
                raise ValueError("fresh frame digest mismatch")
        image_path = directory / source[0]["file"]
        image = read_bgr(image_path)
        if image is None or list(image.shape) != source[0]["shape"] or list(image.shape[1::-1]) != manifest["image_size"]:
            raise ValueError("fresh source dimensions mismatch")
        geometry = {**position, "actions": meta["actions"], "error": f"card_box={tuple(meta['card_box'])}"}
        box = recover_box(geometry, image.shape, interaction)
        if list(box) != item["card_box"]:
            raise ValueError("fresh manifest rectangle mismatch")
        key = (scene, item["row"], item["slot"])
        if key in seen:
            raise ValueError("duplicate fresh scene/slot")
        seen.add(key)
        scenes.add(scene)
        cases.append({
            "transaction": directory.name, "scene_group": scene, "image_path": str(image_path),
            "frame_sha256": sha(image_path), "metadata_sha256": sha(evidence_path),
            "detail_image_path": str(directory / opened[0]["file"]), "card_id": identity,
            "name": names[identity], "box": box, "slot_index": item["slot"] - 1, "plan": item["plan"],
            "partition": "fresh_scene_holdout", "captured_at": meta["captured_at"],
            "truth_basis": "source/detail PNG association reviewed; native title/body and original input retained",
        })
    if not cases or len(scenes) != manifest["scene_count"]:
        raise ValueError("fresh scene count mismatch")
    return cases, []


def freeze_manual_cases(manifest_path, cards):
    """Keep explicit human slot truth distinct from logged input transactions."""
    manifest = read(manifest_path)
    if manifest["truth_basis"] != "user_confirmed_slots_with_detail_images":
        raise ValueError("manual slot truth must be explicit")
    names = {card["id"]: card["name"] for card in cards}
    for frame in manifest["frames"]:
        if sha(frame["path"]).lower() != frame["sha256"].lower():
            raise ValueError("manual evidence image digest mismatch")
    sources = {f["path"]: f for f in manifest["frames"] if f["role"] == "source"}
    if not any(f["role"] == "detail" for f in manifest["frames"]):
        raise ValueError("manual detail images missing")
    seen, cases, source_scenes = set(), [], {}
    for item in manifest["cases"]:
        frame = sources[item["image_path"]]
        previous_scene = source_scenes.setdefault(item["image_path"], item["scene_group"])
        if previous_scene != item["scene_group"]:
            raise ValueError("one manual source image cannot be split across scenes")
        image = read_bgr(item["image_path"])
        if image is None or list(image.shape[1::-1]) != frame["image_size"]:
            raise ValueError("manual source image dimensions mismatch")
        if names[item["card_id"]] != item["name"] or item["plan"] not in {"sense", "logic", "anomaly"}:
            raise ValueError("manual source label mismatch")
        cx, cy, cw, ch = frame["client_box"]
        x, y, w, h = item["box"]
        ih, iw = image.shape[:2]
        if not (0 <= cx <= x < x + w <= cx + cw <= iw and 0 <= cy <= y < y + h <= cy + ch <= ih):
            raise ValueError("manual card box outside reviewed client area")
        key = (item["scene_group"], item["row"], item["slot_index"])
        if key in seen or not 0 <= item["slot_index"] <= 5 or item["row"] not in {0, 1}:
            raise ValueError("manual scene/slot duplicated or invalid")
        seen.add(key)
        cases.append({**item, "partition": "manual_scene_holdout", "frame_sha256": frame["sha256"],
                      "metadata_sha256": sha(manifest_path), "truth_basis": manifest["truth_basis"],
                      "client_box": frame["client_box"]})
    if not cases or len({c["scene_group"] for c in cases}) != manifest["scene_count"]:
        raise ValueError("manual scene count mismatch")
    return cases, []


def ranking(gallery, query, scope, truth):
    started = time.perf_counter()
    hits = gallery.search(query, top_k=5, eligible_card_ids=scope)
    elapsed = (time.perf_counter() - started) * 1000
    all_hits = gallery.search(query, top_k=450, eligible_card_ids=scope)
    rank = next((i + 1 for i, h in enumerate(all_hits) if str(truth) in h.candidate_card_ids), None)
    true_hit = next((h for h in all_hits if str(truth) in h.candidate_card_ids), None)
    return {
        "true_rank": rank,
        "top5_recall": rank is not None and rank <= 5,
        "true_score": None if true_hit is None else true_hit.similarity,
        "top5": [{"ids": list(h.candidate_card_ids), "score": h.similarity, "class_name": h.class_name} for h in hits],
        "search_ms": elapsed,
    }


def run(args):
    root = args.runtime_root
    sys.path.insert(0, str(root / "agent"))
    from arena_winrate.catalog import ArenaEntityCatalog
    from card_selection.embedding import EmbeddingGallery, EmbeddingCardRecognizer

    reader_path = root / "agent/custom/action/arena_reader.py"
    interaction = interaction_function(reader_path.read_text(encoding="utf-8"))
    data = root / "assets/arena-winrate/node_modules/gakumas-data/json"
    cards = read(data / "skill_cards.json")
    catalog = ArenaEntityCatalog(cards, read(data / "customizations.json"))
    manual_manifest = getattr(args, "manual_manifest", None)
    fresh = args.fresh_manifest is not None or manual_manifest is not None
    if manual_manifest:
        cases, excluded = freeze_manual_cases(manual_manifest, cards)
    elif args.fresh_manifest:
        cases, excluded = freeze_fresh_cases(args.fresh_manifest, cards, interaction)
    else:
        if not args.legacy_review:
            raise ValueError("saved transactions require --legacy-review with reviewed source/detail associations")
        cases, excluded = freeze_cases(args.evidence_root, cards, interaction, read(args.legacy_review))
    # Freeze the split before evaluating either encoder or experimental gallery.
    plan = {
        "scope": "OFFLINE_DIAGNOSTIC_ONLY",
        "development_ids": [] if fresh else [99, 752],
        "split_rule": ("all new scenes held out as groups; no tuning or screenshot-derived references" if fresh
                       else "all other confirmed identities held out; repeated frames are not independent samples"),
        "candidate_policy": "arena_only_full_ui_base_unchanged" if args.arena_only else "both_models_full_ui",
        "fresh_manifest_sha256": sha(args.fresh_manifest) if args.fresh_manifest else None,
        "manual_evidence_manifest": read(manual_manifest) if manual_manifest else None,
        "manual_manifest_sha256": sha(manual_manifest) if manual_manifest else None,
        "legacy_review_sha256": sha(args.legacy_review) if getattr(args, "legacy_review", None) else None,
        "cases": cases,
        "excluded": excluded,
        "reader_sha256": sha(reader_path),
        "limitations": [
            "saved source PNG is not the complete original three-frame generation",
            "truth from existing title records, no new OCR",
            "no independent live unknown-set calibration",
            "no screenshot-derived training or references",
            "evidence root can contain migrated historical transactions; path does not prove capture version/date",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    split_path = args.output.with_suffix(".split.json")
    if args.output.exists() or split_path.exists():
        raise FileExistsError("choose a new output name; existing frozen diagnostics must not be overwritten")
    split_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        **plan,
        "split_sha256": sha(split_path),
        "components": {},
        "promotion": "NOT_PROMOTED",
        "runtime": {
            "root": str(root),
            "source_revision": read(root / "GAKUMAS_HELPER_BUILD.json")["source"]["revision"],
            "embedding_source_sha256": sha(root / "agent/card_selection/embedding.py"),
            "geometry_source_sha256": sha(root / "agent/arena_winrate/geometry.py"),
            "catalog_sha256": sha(data / "skill_cards.json"),
        },
    }
    for component in ("embedding/arena_card", "classify/card_embedding"):
        recognizer = EmbeddingCardRecognizer.load(root / "resource/base/model" / component)
        original = recognizer.gallery
        started = time.perf_counter()
        candidate, receipts = original, []
        if not args.arena_only or component == "embedding/arena_card":
            paths, receipts = load_reference_paths(args.dataset_manifest, original.class_names)
            vectors = []
            for path in paths:
                image = read_bgr(path)
                vectors.append(recognizer.embedder.embed(image, (0, 0, image.shape[1], image.shape[0])))
            rendered = np.stack(vectors)
            candidate = EmbeddingGallery(
                np.concatenate((original.embeddings, rendered)),
                (*original.class_names, *(n + "_full_ui" for n in original.class_names)),
                (*original.card_ids, *original.card_ids),
                model_sha256=original.model_sha256,
                visual_group_ids=(*original.visual_group_ids, *original.visual_group_ids),
                upgrade_counts=(*original.upgrade_counts, *original.upgrade_counts),
            )
        results = []
        for case in cases:
            image = read_bgr(case["image_path"])
            scope = catalog.arena_skill_card_candidates(plan=case["plan"], slot_index=case["slot_index"])
            query_started = time.perf_counter()
            query = recognizer.embedder.embed(image, tuple(case["box"]))
            query_ms = (time.perf_counter() - query_started) * 1000
            variants = {}
            x, y, w, h = case["box"]
            if case["card_id"] == 752 and not manual_manifest:
                for dy in (-2, 2):
                    q = recognizer.embedder.embed(image, (x, y + dy, w, h))
                    variants[str(dy)] = ranking(original, q, scope, case["card_id"])
            results.append(
                {
                    "transaction": case["transaction"],
                    "card_id": case["card_id"],
                    "partition": case["partition"],
                    "eligible": case["card_id"] in scope,
                    "query_ms": query_ms,
                    "baseline": ranking(original, query, scope, case["card_id"]),
                    "full_ui_union": ranking(candidate, query, scope, case["card_id"]),
                    "y_sensitivity": variants,
                }
            )
        # Unknown controls show candidate scores, not a calibrated false-accept rate.
        unknown = []
        for value in (0, 127, 255):
            image = np.full((84, 82, 3), value, dtype=np.uint8)
            q = recognizer.embedder.embed(image, (0, 0, 82, 84))
            scope = catalog.arena_skill_card_candidates(plan="sense", slot_index=4)
            unknown.append(
                {"constant_pixel": value, "baseline": ranking(original, q, scope, -1), "full_ui_union": ranking(candidate, q, scope, -1)}
            )
        report["components"][component] = {
            "model_sha256": original.model_sha256,
            "gallery_sha256": original.gallery_sha256,
            "frozen_reference_sources": receipts,
            "original_rows": len(original.card_ids),
            "candidate_rows": len(candidate.card_ids),
            "candidate_changed": candidate is not original,
            "results": results,
            "unknown_controls": unknown,
            "encoding_and_evaluation_seconds": time.perf_counter() - started,
        }
        print(component, "complete", len(results), "transactions", flush=True)
    for i, case in enumerate(cases):
        rows = [c["results"][i] for c in report["components"].values()]
        case["baseline_union_top5"] = any(r["baseline"]["top5_recall"] for r in rows)
        case["candidate_union_top5"] = any(r["full_ui_union"]["top5_recall"] for r in rows)
    report["summary"] = {
        part: {
            "transactions": len(rows),
            "scene_groups": sorted({r["scene_group"] for r in rows if "scene_group" in r}),
            "unique_ids": sorted({r["card_id"] for r in rows}),
            "baseline_recalled": sum(r["baseline_union_top5"] for r in rows),
            "candidate_recalled": sum(r["candidate_union_top5"] for r in rows),
            "regressions": [r["transaction"] for r in rows if r["baseline_union_top5"] and not r["candidate_union_top5"]],
        }
        for part in ("development", "heldout_other_identity", "fresh_scene_holdout", "manual_scene_holdout")
        if (rows := [r for r in cases if r["partition"] == part])
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--evidence-root", type=Path)
    evidence.add_argument("--fresh-manifest", type=Path)
    evidence.add_argument("--manual-manifest", type=Path, help="Explicit user slot labels; never treated as logged clicks")
    parser.add_argument("--arena-only", action="store_true", help="Keep the base gallery unchanged")
    parser.add_argument("--legacy-review", type=Path, help="Private transaction review required with --evidence-root")
    parser.add_argument("--dataset-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
