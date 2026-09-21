"""Expiring, bounded capture of observations the reader already made.

No controller, recognizer or identity decision is available to this module.
Probe samples share the existing reader-failure export directory, but carry a
different event and explicitly unreviewed labels.
"""
from __future__ import annotations

import json
import shutil
from typing import Any
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from collections.abc import Mapping

CAMPAIGN = "arena-feedback-20260920"
EXPIRES_AT = "2026-11-01T00:00:00+08:00"


def release_probe_settings(package_root: Path | None = None) -> dict:
    """Enable the approved campaign in installed packages, not development trees."""
    root = package_root or Path(__file__).resolve().parents[2]
    path = root / "GAKUMAS_HELPER_BUILD.json"
    try:
        if not path.is_file() or path.stat().st_size > 256 * 1024:
            return {}
        build = json.loads(path.read_text(encoding="utf-8"))
        if build.get("product") != "MaaGakumasu":
            return {}
        return {"enabled": True, "campaign": CAMPAIGN,
                "starts_at": "2026-09-20T00:00:00+08:00", "expires_at": EXPIRES_AT,
                "max_samples": 20, "max_bytes": 128 * 1024 * 1024,
                "build": {"version": build.get("derived_version"),
                          "release_source": build.get("source", {}).get("revision")}}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


class RecognitionProbe:
    TARGETS = frozenset({297, 299, 389, 553, 752})
    MAX_SAMPLES = 20
    MAX_BYTES = 128 * 1024 * 1024

    def __init__(self, root: Path):
        self.root = root
        self.settings: dict[str, Any] = release_probe_settings()
        self.disabled = False
        try:
            path = root / "recognition-probe.json"
            if path.is_file():
                self.settings = {}
                if path.stat().st_size <= 8192:
                    self.settings = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(self.settings, dict):
                        self.settings = {}
        except (OSError, ValueError, TypeError):
            self.disabled = True

    def active(self) -> bool:
        try:
            now = datetime.now(timezone.utc)
            return bool(
                not self.disabled and self.settings.get("enabled") is True
                and datetime.fromisoformat(self.settings["starts_at"]) <= now
                < datetime.fromisoformat(self.settings["expires_at"])
            )
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def observe(diagnostic: dict, image: Any, items: Any, phase: str, *, reco_id=None, seconds=None) -> None:
        frames = diagnostic.get("probe_frames")
        if frames is None or any(frame["image"] is image for frame in frames):
            return
        raw = []
        for item in items:
            value = item.get if isinstance(item, Mapping) else lambda key, default: getattr(item, key, default)
            box = value("box", ()) or ()
            raw.append({"text": str(value("text", "")),
                        "box": [int(value) for value in box],
                        "score": float(value("score", 0) or 0)})
        frames.append({"image": image, "ocr": raw, "phase": phase, "reco_id": reco_id, "seconds": seconds})
        if len(frames) > 6:
            # Preserve the first two observations and four most recent ones.
            del frames[2]

    def save(self, diagnostic: dict, outcome: str, error: str | None) -> dict | None:
        if not self.active() or "probe_sources" not in diagnostic:
            return None
        position = diagnostic["position"]
        card_id = position.get("observed_detail_card_id")
        p_item = position.get("kind") == "p_item"
        if p_item and outcome != "completed" and "p_item_source_restore_unproven" not in str(error):
            return None
        target = card_id if card_id in self.TARGETS else None
        # Non-target samples are controls, never automatically "unknown cards".
        category = ("p_item_return_control" if outcome == "completed" else "p_item_return_failed") if p_item else (str(target) if target else "control")
        key = [position.get(name) for name in (
            "team_id", "stage_number", "member_slot", "group_index", "card_slot",
        )] + [card_id, outcome]
        if p_item:
            key.append(position.get("screen_slot"))
        folder = self.root / "reader-failures"
        records = []
        used = 0
        for sample in folder.glob("probe-*"):
            if sample.is_dir():
                used += sum(path.stat().st_size for path in sample.iterdir() if path.is_file())
                metadata = sample / "evidence.json"
                if metadata.exists():
                    records.append(json.loads(metadata.read_text(encoding="utf-8")))
        max_samples = min(self.MAX_SAMPLES, int(self.settings.get("max_samples", self.MAX_SAMPLES)))
        max_bytes = min(self.MAX_BYTES, int(self.settings.get("max_bytes", self.MAX_BYTES)))
        counts = Counter(record.get("probe_category") for record in records)
        # Reserve four of the shared twenty samples for the return investigation.
        if not p_item and sum(v for k, v in counts.items() if not str(k).startswith("p_item_return_")) >= max_samples - min(4, max_samples // 5):
            return None
        if len(records) >= max_samples or used >= max_bytes:
            self.disabled = True
            return None
        category_limit = (1 if outcome == "completed" else 3) if p_item else (3 if target else 5)
        if (counts[category] >= category_limit
                or any(record.get("probe_key") == key for record in records)):
            return None
        if shutil.disk_usage(self.root).free < max_bytes + 256 * 1024 * 1024:
            return None
        import cv2

        images = []
        for index, image in enumerate(diagnostic["probe_sources"]):
            images.append((f"source-{index + 1}", image, None))
        for frame in diagnostic["probe_frames"]:
            if not any(image is frame["image"] for _, image, _ in images):
                images.append(("observed-" + frame["phase"], frame["image"], frame["ocr"]))
        confirmed = diagnostic.get("confirmed_open")
        if confirmed is not None and not any(image is confirmed[1] for _, image, _ in images):
            images.append(("title-confirmed-open", confirmed[1], None))
        if not images:
            return None
        encoded_frames = []
        saved = []
        for index, (role, image, ocr) in enumerate(images[:10]):
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("probe PNG encoding failed")
            data = encoded.tobytes()
            encoded_frames.append(data)
            saved.append({"file": f"{index:02d}.png", "role": role,
                          "shape": list(image.shape), "ocr": ocr})
        evidence = {
            "event": "arena_recognition_probe_sample", **position,
            "campaign": self.settings.get("campaign", CAMPAIGN),
            "expires_at": self.settings.get("expires_at"),
            "transaction_id": diagnostic["transaction_id"],
            "member_read_id": diagnostic.get("member_read_id"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "outcome": outcome, "error": error, "actions": list(diagnostic["actions"]),
            "frames": saved, "probe_category": category, "probe_key": key,
            "observations": [{"file": f"{index:02d}.png", "phase": frame["phase"],
                              "reco_id": frame.get("reco_id"), "seconds": frame.get("seconds")}
                             for index, (_, img, _) in enumerate(images[:10])
                             for frame in diagnostic["probe_frames"] if frame["image"] is img],
            "source_frame_count": len(diagnostic["probe_sources"]),
            "source_frames_distinct_objects": len({id(image) for image in diagnostic["probe_sources"]}),
            "source_card_box": diagnostic.get("probe_source_box"),
            "source_row_boxes": diagnostic.get("probe_row_boxes"),
            "visual_candidates": diagnostic.get("probe_candidates"),
            "stage_plan": diagnostic.get("probe_stage_plan"),
            "body_text": diagnostic.get("probe_body_text"),
            "p_item_restore": diagnostic.get("p_item_restore"),
            "reader_probe_revision": "arena-feedback-20260920-v1",
            "build": self.settings.get("build"),
            "truth_status": "unreviewed_observation_not_training_label",
            "extra_screenshots": 0, "extra_ocr": 0, "extra_clicks": 0,
        }
        metadata = json.dumps(evidence, ensure_ascii=False, indent=2).encode("utf-8")
        if used + sum(map(len, encoded_frames)) + len(metadata) > max_bytes:
            return None
        destination = folder / ("probe-" + diagnostic["transaction_id"])
        destination.mkdir(parents=True, exist_ok=False)
        # Write metadata last: incomplete files cannot masquerade as a complete pair.
        for frame, data in zip(saved, encoded_frames, strict=True):
            (destination / frame["file"]).write_bytes(data)
        (destination / "evidence.json").write_bytes(metadata)
        return {"event": "arena_recognition_probe_saved", "folder": str(destination),
                "card_id": card_id, "outcome": outcome, "frames": len(saved)}
