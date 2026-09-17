"""Offline sampling comparison; no production assets, capture, input, or training."""

from __future__ import annotations

import sys
import json
import time
import hashlib
import argparse
from pathlib import Path

import cv2
import numpy as np

BASE_REFERENCE_DIGEST = "22391DA047D1C223B47B9FD292391422F26D7C5AF76950A8D9E4327E4564EAB1"
VARIANTS = ("top50", "top58", "top62", "notch", "notch_ui_guarded")
WEIGHTS = (0.05, 0.10, 0.15, 0.20)
SUPPLEMENT_INDICES = {"A": tuple(range(128, 144)), "B": tuple(range(128, 149))}
WEIGHTED_CONFIGS = {f"{region}_{weight:.2f}": (region, weight)
                    for region in SUPPLEMENT_INDICES for weight in WEIGHTS}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def decode(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def normalize(image, box=None):
    if box is not None:
        x, y, w, h = box
        if min(x, y) < 0 or min(w, h) < 8:
            raise ValueError("Invalid ROI")
        image = image[y:y+h, x:x+w, :3]
        if image.shape[:2] != (h, w):
            raise ValueError("ROI outside image")
    return cv2.resize(np.ascontiguousarray(image), (96, 96), interpolation=cv2.INTER_AREA)


def notch_indices(guard_ui=False):
    # Whole 6x6 cells only: excluded pixels cannot bleed into a selected cell.
    keep = np.ones((16, 16), dtype=bool)
    exclusions = [(30, 54, 66, 96)]
    if guard_ui:
        # Existing embedding lower-corner exclusions, conservatively rounded out.
        exclusions += [(0, 61.44, 30.72, 96), (61.44, 55.68, 96, 96)]
    for row in range(16):
        for col in range(16):
            x, y = col * 6, row * 6
            if any(x < r and x + 6 > l and y < b and y + 6 > t
                   for l, t, r, b in exclusions):
                keep[row, col] = False
    return np.flatnonzero(keep.ravel())


MASKS = {"notch": notch_indices(), "notch_ui_guarded": notch_indices(True)}


def feature(normalized, variant):
    if variant in MASKS:
        cells = cv2.resize(normalized, (16, 16), interpolation=cv2.INTER_AREA)
        return cells.reshape(-1, 3)[MASKS[variant]].astype(np.int16)
    height, rows = {"top50": (48, 8), "top58": (56, 9), "top62": (60, 10)}[variant]
    return cv2.resize(normalized[:height], (16, rows), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.int16)


def page_and_identity_features(normalized, region):
    """Keep the full page descriptor; identity sampling is a separate view."""
    page = cv2.resize(normalized, (16, 16), interpolation=cv2.INTER_AREA)
    indices = (*range(128), *SUPPLEMENT_INDICES[region])
    return page, page.reshape(-1, 3)[list(indices)].astype(np.int16)


def weighted_reference_errors(references, query, weight):
    """One current-frame/gallery difference, never a frame/frame difference."""
    if not 0 <= weight <= 1 or query.shape[0] not in (144, 149):
        raise ValueError("Invalid weighted sampling configuration")
    difference = np.abs(references - query)
    legacy = np.mean(difference[:, :128], axis=(1, 2))
    extra = np.mean(difference[:, 128:], axis=(1, 2))
    return legacy, (1 - weight) * legacy + weight * extra


def outcome_changes(old, new, truth):
    wrong = len(new) == 1 and new != [truth]
    return {"wrong_unique": wrong, "new_wrong_unique": wrong and new != old,
            "lost_correct_unique": old == [truth] and new != [truth],
            "correct_unique_improvement": old != [truth] and new == [truth],
            "wrong_to_inconclusive": len(old) == 1 and old != [truth] and len(new) != 1}


def grouped_case_summary(rows, config):
    """A crop variant cannot count as another independent success."""
    groups = {}
    for row in rows:
        groups.setdefault(row["sample_unit"], []).append(row)
    totals = {"sample_units": len(groups), "wrong_unique": 0, "new_wrong_unique": 0,
              "lost_correct_unique": 0, "ordinary_correct_unique_improvements": 0,
              "wrong_to_inconclusive": 0}
    for unit in groups.values():
        changes = [outcome_changes(candidates(r["results"]["top50"]),
                                   candidates(r["results"][config]), r["truth"]) for r in unit]
        for key in ("wrong_unique", "new_wrong_unique", "lost_correct_unique", "wrong_to_inconclusive"):
            totals[key] += int(any(c[key] for c in changes))
        # Improvement must survive every recorded view of the same original.
        totals["ordinary_correct_unique_improvements"] += int(
            all(r["ordinary_counterexample"] for r in unit)
            and any(c["correct_unique_improvement"] for c in changes)
            and all(candidates(r["results"][config]) == [r["truth"]] for r in unit))
    return totals


def candidate_rank(summary, p95, region, weight):
    return (summary["wrong_unique"], -summary["ordinary_correct_unique_improvements"],
            p95, len(SUPPLEMENT_INDICES[region]), weight)


def candidates(result):
    items = result.get("candidates", ()) if result["status"] == "MEASURED_CANDIDATES" else (result,)
    return sorted({item["reference_business_id"] for item in items
                   if item.get("status") == "MEASURED" and not item.get("identity_low_confidence", True)})


def geometry_boxes(box, shape):
    """Existing evaluator's 41 non-original offsets; omit unavailable pixels."""
    x, y, w, h = box
    for dx in range(-3, 4):
        for dy in range(-1, 2):
            for extra in (0, 2):
                if (dx, dy, extra) == (0, 0, 0):
                    continue
                b = (x + dx, y + dy, w + extra, h + extra)
                if b[0] >= 0 and b[1] >= 0 and b[0] + b[2] <= shape[1] and b[1] + b[3] <= shape[0]:
                    yield b


def load_sources(args, gallery):
    extensions = {}
    for path in args.extension_manifest:
        manifest = read(path)
        for item in manifest["icons"]["images"]:
            source = path.parent / manifest["icons"]["root"] / item["path"]
            if digest(source) != item["sha256"].upper():
                raise ValueError("Extension source hash mismatch")
            if source.name in extensions:
                raise ValueError("Duplicate extension name")
            extensions[source.name] = source
    live_manifest = read(args.live_manifest)
    if len(live_manifest["samples"]) != 1:
        raise ValueError("This diagnostic supports the current single live prototype only")
    live = live_manifest["samples"][0]
    images, receipts = [], []
    initial = hashlib.sha256()
    for i, name in enumerate(gallery.reference_names):
        if name == "L0001":
            source = args.live_manifest.parent / live["image_path"]
            if digest(source) != live["image_sha256"].upper():
                raise ValueError("Live source hash mismatch")
            normalized = normalize(decode(source), live["roi"])
        else:
            source = extensions.get(name, args.reference_root / name)
            normalized = normalize(decode(source))
        sha = digest(source)
        if i < 3388:
            initial.update(name.encode())
            initial.update(b"\0")
            initial.update(bytes.fromhex(sha))
        if not np.array_equal(feature(normalized, "top50"), gallery.top_coarse[i].reshape(-1, 3)):
            raise ValueError(f"Production baseline pixel mismatch: {name}")
        images.append(normalized)
        receipts.append({"reference": name, "path": str(source.resolve()), "sha256": sha})
    if initial.hexdigest().upper() != BASE_REFERENCE_DIGEST:
        raise ValueError("Original 3388 full-PNG source digest mismatch")
    return images, receipts


def compare(args):
    sys.path.insert(0, str(args.runtime_root / "agent"))
    from arena_winrate.catalog import ArenaEntityCatalog
    from arena_winrate.badge_reference import BadgeReferenceGallery

    output = args.output
    if output.exists() or output.with_suffix(".split.json").exists():
        raise FileExistsError("Refuse to overwrite a frozen diagnostic")
    gallery_root = args.runtime_root / "resource/base/model/embedding/arena_badge_reference"
    gallery = BadgeReferenceGallery.load(gallery_root)
    data = args.runtime_root / "assets/arena-winrate/node_modules/gakumas-data/json"
    catalog = ArenaEntityCatalog(read(data / "skill_cards.json"), read(data / "customizations.json"))
    images, receipts = load_sources(args, gallery)
    cases = read(args.cases)["cases"]
    for case in cases:
        if digest(case["image_path"]) != case["image_sha256"].upper():
            raise ValueError("Case image changed")
    scopes = {(p, s): catalog.arena_skill_card_candidates(plan=p, slot_index=s)
              for p in ("sense", "logic", "anomaly") for s in (0, 1, 2)}
    split = {"variants": list(VARIANTS), "mask_indices": {k: v.tolist() for k, v in MASKS.items()},
             "policy": "fixed pre-scoring; current thresholds; one distance pass; no model",
             "cases": cases, "source_receipts": receipts,
             "gallery_manifest_sha256": digest(gallery_root / "manifest.json"),
             "runtime_revision": read(args.runtime_root / "GAKUMAS_HELPER_BUILD.json")["source"]["revision"]}
    split["geometry_policy"] = {"dx": list(range(-3, 4)), "dy": [-1, 0, 1],
                                "extra_size": [0, 2], "out_of_bounds": "omit_no_padding"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    matrices = {v: np.stack([feature(im, v) for im in images]) for v in VARIANTS}
    scoped = {}
    for key, scope in scopes.items():
        indices, _, _ = gallery._eligible_scope(scope)
        scoped[key] = {v: matrices[v][indices] for v in VARIANTS}

    def score(normalized, variant, key):
        query = feature(normalized, variant)
        errors = np.mean(np.abs(scoped[key][variant] - query), axis=(1, 2))
        return gallery._identity_result_from_errors(errors, scopes[key])

    results = []
    for case in cases:
        normalized = normalize(decode(case["image_path"]), case["box"])
        key = (case["plan"], min(2, case["slot_index"]))
        scores = {v: score(normalized, v, key) for v in VARIANTS}
        original = gallery.measure_identity(normalized, (0, 0, 96, 96), eligible_card_ids=scopes[key])
        if scores["top50"] != original:
            raise AssertionError("Baseline decision differs from production")
        results.append({**case, "results": scores})
    print("Frozen live cases scored", len(results), flush=True)

    if args.geometry_only:
        geometry = []
        for case in cases:
            im = decode(case["image_path"])
            key = (case["plan"], min(2, case["slot_index"]))
            for box in geometry_boxes(case["box"], im.shape):
                normalized = normalize(im, box)
                measurements = {v: score(normalized, v, key) for v in VARIANTS}
                geometry.append({"case": case["case"], "truth": case["truth"],
                                 "box": box, "results": measurements})
        summary = {}
        for v in VARIANTS:
            wrong, lost, new_wrong = [], [], []
            for row in geometry:
                old = candidates(row["results"]["top50"])
                new = candidates(row["results"][v])
                brief = {"case": row["case"], "box": row["box"], "truth": row["truth"],
                         "baseline": old, "candidate": new}
                if len(new) == 1 and row["truth"] not in new:
                    wrong.append(brief)
                    if new != old:
                        new_wrong.append(brief)
                if old == [row["truth"]] and new != old:
                    lost.append(brief)
            summary[v] = {"wrong_unique": wrong, "new_wrong_unique": new_wrong,
                          "lost_correct_unique": lost}
        output.write_text(json.dumps({"status": "GEOMETRY_DIAGNOSTIC_ONLY_NOT_PROMOTED",
                                      "sample_count": len(geometry), "summary": summary,
                                      "geometry": geometry}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"geometry_samples": len(geometry), "summary": {
            v: {k: len(x) for k, x in s.items()} for v, s in summary.items()}}))
        return

    # Full same-input source regression over all nine legal plan/slot scopes.
    # This is a static reference guard, not independent gameplay accuracy.
    regressions = {v: {"queries": 0, "wrong_unique": 0, "truth_in_candidates": 0,
                       "new_wrong_unique": [], "lost_correct_unique": []} for v in VARIANTS}
    for key, scope in scopes.items():
        count = 0
        for i, im in enumerate(images):
            truth = int(gallery.business_ids[i])
            if truth not in scope:
                continue
            found = {v: candidates(score(im, v, key)) for v in VARIANTS}
            for v in VARIANTS:
                stat = regressions[v]
                stat["queries"] += 1
                stat["wrong_unique"] += int(len(found[v]) == 1 and truth not in found[v])
                stat["truth_in_candidates"] += int(truth in found[v])
                marker = {"plan": key[0], "slot_index": key[1], "reference": gallery.reference_names[i],
                          "truth": truth, "baseline": found["top50"], "candidate": found[v]}
                if len(found[v]) == 1 and truth not in found[v] and found[v] != found["top50"]:
                    stat["new_wrong_unique"].append(marker)
                if found["top50"] == [truth] and found[v] != [truth]:
                    stat["lost_correct_unique"].append(marker)
            count += 1
        print("Static scope complete", key, count, flush=True)

    # Interleaved full feature+distance+ranking timings, no inference or I/O.
    timings = {v: [] for v in VARIANTS}
    timing_images = []
    for case in cases:
        image = decode(case["image_path"])
        x, y, w, h = case["box"]
        timing_images.append((image[y:y+h, x:x+w],
                              (case["plan"], min(2, case["slot_index"]))))
    for iteration in range(120):
        for im, key in timing_images:
            order = VARIANTS[iteration % len(VARIANTS):] + VARIANTS[:iteration % len(VARIANTS)]
            for v in order:
                started = time.perf_counter_ns()
                score(normalize(im), v, key)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if iteration >= 20:
                    timings[v].append(elapsed)
    summary = {v: {"points": int(matrices[v].shape[1]), "measurements": len(timings[v]),
                   "p50_ms": float(np.percentile(timings[v], 50)),
                   "p95_ms": float(np.percentile(timings[v], 95))} for v in VARIANTS}
    report = {"status": "OFFLINE_DIAGNOSTIC_ONLY_NOT_PROMOTED", "cases": results,
              "source_rows": len(images), "baseline_all_rows_bitwise_equal": True,
              "baseline_original_png_digest_verified": True,
              "reference_regression": regressions, "compute_timing": summary,
              "limitations": ["No own-team wall-time test; <=2% gate remains pending",
                              "Static source regression is not independent gameplay accuracy",
                              "Old known cases are development evidence; no parameter retuning in this run"]}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"compute_timing": summary, "report": str(output)}, ensure_ascii=False))


def compare_weighted(args):
    """Frozen eight-candidate experiment. Controls never enter promotion ranking."""
    sys.path.insert(0, str(args.runtime_root / "agent"))
    from arena_winrate.catalog import ArenaEntityCatalog
    from arena_winrate.badge_reference import BadgeReferenceGallery

    output = args.output
    split_path = output.with_suffix(".split.json")
    if output.exists() or split_path.exists():
        raise FileExistsError("Refuse to overwrite a frozen diagnostic")
    root = args.runtime_root / "resource/base/model/embedding/arena_badge_reference"
    runtime_files = [root / "manifest.json", root / "badge_reference_gallery.npz",
                     args.runtime_root / "agent/arena_winrate/badge_reference.py",
                     args.runtime_root / "agent/custom/action/arena_reader.py"]
    frozen_runtime = {str(p): digest(p) for p in runtime_files}
    gallery = BadgeReferenceGallery.load(root)
    data = args.runtime_root / "assets/arena-winrate/node_modules/gakumas-data/json"
    catalog = ArenaEntityCatalog(read(data / "skill_cards.json"), read(data / "customizations.json"))
    images, receipts = load_sources(args, gallery)
    cases = read(args.cases)["cases"]
    for case in cases:
        if digest(case["image_path"]) != case["image_sha256"].upper():
            raise ValueError("Case image changed")
        if not case.get("sample_unit") or type(case.get("ordinary_counterexample")) is not bool:
            raise ValueError("Weighted cases require explicit sample_unit and ordinary_counterexample")
    scopes = {(p, s): catalog.arena_skill_card_candidates(plan=p, slot_index=s)
              for p in ("sense", "logic", "anomaly") for s in (0, 1, 2)}
    configurations = ("top50", "top62", *WEIGHTED_CONFIGS)
    matrices = {v: np.stack([feature(im, v) for im in images]) for v in ("top50", "top62")}
    for region in SUPPLEMENT_INDICES:
        matrices[region] = np.stack([page_and_identity_features(im, region)[1] for im in images])
        np.testing.assert_array_equal(matrices[region][:, :128], matrices["top50"])
    scoped = {}
    for key, scope in scopes.items():
        indices, _, _ = gallery._eligible_scope(scope)
        scoped[key] = {v: matrix[indices] for v, matrix in matrices.items()}
    split = {"revision": "weighted-supplement-v1", "candidate_configs": WEIGHTED_CONFIGS,
             "controls": ["top50", "top62", "A_lambda0", "B_lambda0"],
             "supplement_indices": SUPPLEMENT_INDICES, "source_receipts": receipts, "cases": cases,
             "case_manifest_sha256": digest(args.cases), "runtime_files": frozen_runtime,
             "evaluator_sha256": digest(Path(__file__)),
             "runtime_revision": read(args.runtime_root / "GAKUMAS_HELPER_BUILD.json")["source"]["revision"],
             "geometry_policy": {"dx": list(range(-3, 4)), "dy": [-1, 0, 1], "extra_size": [0, 2],
                                 "out_of_bounds": "omit_no_padding"},
             "thresholds": {"maximum_coarse_error": gallery.maximum_coarse_error,
                            "maximum_zero_coarse_error": gallery.maximum_zero_coarse_error,
                            "minimum_group_margin": gallery.minimum_group_margin,
                            "minimum_high_error_group_margin": gallery.minimum_high_error_group_margin},
             "timing": {"warmups_per_config": 20, "measured_queries_per_config": 1000},
             "selection": "wrong_unique,-ordinary_improvements,p95,extra_points,weight",
             "sample_policy": "all views in a sample_unit must become correct to count one ordinary improvement"}
    output.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    def score(normalized, config, key):
        if config in WEIGHTED_CONFIGS:
            region, weight = WEIGHTED_CONFIGS[config]
            page, query = page_and_identity_features(normalized, region)
            legacy, errors = weighted_reference_errors(scoped[key][region], query, weight)
        else:
            # Production already constructs the full page descriptor as well.
            page = cv2.resize(normalized, (16, 16), interpolation=cv2.INTER_AREA)
            query = feature(normalized, config)
            difference = np.abs(scoped[key][config] - query)
            errors = np.mean(difference, axis=(1, 2))
            legacy = errors if config == "top50" else np.mean(difference[:, :128], axis=(1, 2))
        old_reference_index = int(np.argmin(legacy))
        return gallery._identity_result_from_errors(errors, scopes[key]), page, old_reference_index

    equivalence_count = 0

    def evaluate(normalized, key):
        nonlocal equivalence_count
        results = {v: score(normalized, v, key)[0] for v in configurations}
        original_group, original_page, original_result = gallery.content_signature_with_identity(
            normalized, (0, 0, 96, 96), eligible_card_ids=scopes[key])
        if original_result != results["top50"]:
            raise AssertionError("Original baseline result differs")
        original_errors = np.mean(np.abs(scoped[key]["top50"] - feature(normalized, "top50")), axis=(1, 2))
        indices, _, _ = gallery._eligible_scope(scopes[key])
        for region in SUPPLEMENT_INDICES:
            page, query = page_and_identity_features(normalized, region)
            old, zero = weighted_reference_errors(scoped[key][region], query, 0)
            np.testing.assert_array_equal(zero, original_errors)
            np.testing.assert_array_equal(page, original_page)
            if gallery.visual_group_ids[int(indices[int(np.argmin(old))])] != original_group:
                raise AssertionError("Legacy page reference changed")
            if gallery._identity_result_from_errors(zero, scopes[key]) != original_result:
                raise AssertionError("Lambda zero changed complete identity result")
        equivalence_count += 1
        return results

    results = []
    for case in cases:
        key = (case["plan"], min(2, case["slot_index"]))
        results.append({**case, "results": evaluate(normalize(decode(case["image_path"]), case["box"]), key)})
    print("Original inputs and lambda-zero comparisons complete", len(results), flush=True)
    static = {v: {"queries": 0, "wrong_unique": 0, "new_wrong_unique": [], "lost_correct_unique": []}
              for v in configurations}
    for key, scope in scopes.items():
        for i, im in enumerate(images):
            truth = int(gallery.business_ids[i])
            if truth not in scope:
                continue
            measured = evaluate(im, key)
            old = candidates(measured["top50"])
            for v in configurations:
                new = candidates(measured[v])
                delta = outcome_changes(old, new, truth)
                stat = static[v]
                stat["queries"] += 1
                stat["wrong_unique"] += int(delta["wrong_unique"])
                for reason in ("new_wrong_unique", "lost_correct_unique"):
                    if delta[reason]:
                        stat[reason].append({"scope": key, "reference": gallery.reference_names[i],
                                             "truth": truth, "baseline": old, "candidate": new})
        print("Weighted static scope complete", key, flush=True)
    geometry = []
    pressure = {v: {"wrong_unique": 0, "new_wrong_unique": 0, "lost_correct_unique": 0} for v in configurations}
    for case in cases:
        im = decode(case["image_path"])
        key = (case["plan"], min(2, case["slot_index"]))
        for box in geometry_boxes(case["box"], im.shape):
            measured = evaluate(normalize(im, box), key)
            geometry.append({"case": case["case"], "sample_unit": case["sample_unit"],
                             "truth": case["truth"], "box": box, "results": measured})
            for v in configurations:
                delta = outcome_changes(candidates(measured["top50"]), candidates(measured[v]), case["truth"])
                for metric in pressure[v]:
                    pressure[v][metric] += int(delta[metric])
    print("Geometry and lambda-zero comparisons complete", len(geometry), flush=True)
    timing_inputs = [(decode(c["image_path"]), c["box"], (c["plan"], min(2, c["slot_index"]))) for c in cases]
    timings = {v: [] for v in configurations}
    for iteration in range(1020):
        im, box, key = timing_inputs[iteration % len(timing_inputs)]
        offset = iteration % len(configurations)
        for config in configurations[offset:] + configurations[:offset]:
            started = time.perf_counter_ns()
            score(normalize(im, box), config, key)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if iteration >= 20:
                timings[config].append(elapsed)
    timing = {v: {"measurements": len(values), "p50_ms": float(np.percentile(values, 50)),
                  "p95_ms": float(np.percentile(values, 95))} for v, values in timings.items()}
    summaries = {v: grouped_case_summary(results, v) for v in configurations}
    rankings = []
    for v, (region, weight) in WEIGHTED_CONFIGS.items():
        failures = []
        for metric in ("new_wrong_unique", "lost_correct_unique"):
            if summaries[v][metric] or pressure[v][metric] or static[v][metric]:
                failures.append(metric)
        if not summaries[v]["ordinary_correct_unique_improvements"]:
            failures.append("no_ordinary_correct_unique_improvement")
        rankings.append({"config": v, "reliability_failures": failures,
                         "rank_key": candidate_rank(summaries[v], timing[v]["p95_ms"], region, weight)})
    qualified_rankings = sorted((r for r in rankings if not r["reliability_failures"]),
                                key=lambda row: row["rank_key"])
    for path, expected in frozen_runtime.items():
        if digest(Path(path)) != expected:
            raise ValueError("Runtime changed during diagnostic; discard this run")
    qualified = [r["config"] for r in qualified_rankings]
    report = {"status": "OFFLINE_ONLY_NOT_PROMOTED", "candidate_count": len(WEIGHTED_CONFIGS),
              "source_rows": len(images), "equivalence_queries": equivalence_count,
              "lambda_zero_complete_results_equal": True, "full_page_and_legacy_reference_equal": True,
              "cases": results, "sample_summary": summaries, "static_reference_regression": static,
              "geometry_count": len(geometry), "geometry_summary": pressure, "geometry": geometry,
              "compute_timing": timing, "candidate_results": rankings,
              "candidate_ranking": qualified_rankings,
              "numerically_qualified_candidates": qualified,
              "next_gate": "independent_validation_and_page_guard_tests" if qualified else "STOP_KEEP_BASELINE",
              "limitations": ["No native three-frame transaction replay in this case manifest; synthetic contract tests are separate",
                              "Source crops and their perturbations are development evidence, not independent scenes",
                              "Static reference queries are not gameplay accuracy",
                              "No measured reduction in clicks, model calls, or full-reading latency",
                              "Own-team P50/P95 <=102% gate not run; opponent and overall arena acceptance pending",
                              "No production, page-protection, threshold, model, or candidate-scope changes"]}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summaries, "geometry": pressure, "timing": timing,
                      "ranking": qualified_rankings, "next_gate": report["next_gate"]}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--extension-manifest", type=Path, action="append", required=True)
    p.add_argument("--live-manifest", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--geometry-only", action="store_true")
    p.add_argument("--weighted", action="store_true", help="Frozen eight-config supplement experiment")
    args = p.parse_args()
    if args.weighted and args.geometry_only:
        p.error("--weighted includes geometry; do not combine with --geometry-only")
    (compare_weighted if args.weighted else compare)(args)


if __name__ == "__main__":
    main()
