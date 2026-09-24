"""Expiring, bounded capture of observations the reader already made.

No controller, recognizer or identity decision is available to this module.
Probe samples share the existing reader-failure export directory, but carry a
different event and explicitly unreviewed labels.
"""
from __future__ import annotations

import re
import json
import stat
import shutil
from typing import Any
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Mapping

from .evidence_retention import _FRAME_NAME, _plain_stat

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
    TARGETS = frozenset({31, 297, 403, 553})
    RETIRED_TARGETS = frozenset({299, 389, 752})
    MAX_SAMPLES = 20
    MAX_BYTES = 128 * 1024 * 1024

    def __init__(self, root: Path):
        self.root = root
        self.settings: dict[str, Any] = release_probe_settings()
        release_build = self.settings.get("build")
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
        if release_build is not None:
            # A retained user override controls capture, not the running version.
            self.settings["build"] = release_build

    @staticmethod
    def _group_files(group: Path, folder: Path) -> list[Path]:
        if group.resolve().parent != folder or not stat.S_ISDIR(_plain_stat(group).st_mode):
            raise OSError("probe group outside evidence directory")
        files = list(group.iterdir())
        for path in files:
            if (path.name != "evidence.json" and not _FRAME_NAME.fullmatch(path.name)) or not stat.S_ISREG(_plain_stat(path).st_mode):
                raise OSError("unexpected probe group contents")
        return files

    @classmethod
    def _remove_group(cls, group: Path, folder: Path) -> None:
        files = cls._group_files(group, folder)
        for path in sorted(files, key=lambda p: p.name == "evidence.json"):
            path.unlink()
        group.rmdir()

    def _records(self, folder: Path) -> list[dict]:
        records = []
        for group in folder.glob("probe-*"):
            if not re.fullmatch(r"probe-[0-9a-f]{32}", group.name):
                continue
            try:
                files = self._group_files(group, folder)
                metadata = group / "evidence.json"
                if metadata.stat().st_size > 1024 * 1024:
                    continue
                value = json.loads(metadata.read_text(encoding="utf-8"))
                if value.get("event") != "arena_recognition_probe_sample":
                    continue
                recorded = datetime.fromisoformat(value["recorded_at"])
                if recorded.tzinfo is None:
                    continue
                records.append({"path": group, "value": value, "at": recorded.timestamp(),
                                "bytes": sum(p.stat().st_size for p in files)})
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
        return sorted(records, key=lambda r: (r["at"], r["path"].name))

    def _evictions(self, records, category, key, size, max_samples, max_bytes):
        remaining = list(records)
        victims = []

        def remove(record):
            remaining.remove(record)
            victims.append(record["path"])

        # Finished investigations and repeated positions cannot crowd out new
        # evidence. A new transaction replaces the prior view of the same slot.
        for record in list(remaining):
            value = record["value"]
            if value.get("probe_category") in {str(x) for x in self.RETIRED_TARGETS} or value.get("probe_key") == key:
                remove(record)
        category_limit = 1 if category == "p_item_return_control" else 3 if category != "control" else 5
        same = [r for r in remaining if r["value"].get("probe_category") == category]
        while len(same) >= category_limit:
            remove(same.pop(0))
        if not category.startswith("p_item_return_"):
            cards = [r for r in remaining if not str(r["value"].get("probe_category", "")).startswith("p_item_return_")]
            card_limit = max_samples - min(4, max_samples // 5)
            while len(cards) >= card_limit:
                remove(cards.pop(0))
        while len(remaining) >= max_samples or sum(r["bytes"] for r in remaining) + size > max_bytes:
            if not remaining:
                return None
            remove(remaining[0])
        return victims

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

    @staticmethod
    def observe_identity_check(diagnostic: dict | None, image: Any, *, card_id: int, contact: int, accepted: bool) -> None:
        """Pin two existing opening observations before the rolling window drops them."""
        if diagnostic is None or "probe_sources" not in diagnostic:
            return
        frames = diagnostic.get("probe_frames", ())
        observation = next((frame for frame in frames if frame["image"] is image), None)
        if observation is None:
            return
        checks = diagnostic.get("probe_identity_checks", ())
        if contact == 1 and not accepted:
            candidates = diagnostic.get("probe_candidates", ())
            if not candidates or card_id in candidates:
                return
            role = "candidate-mismatch-open"
        elif contact == 2 and accepted and checks:
            role = "candidate-reopen-confirmed"
        else:
            return
        if any(check["role"] == role for check in checks):
            return
        diagnostic.setdefault("probe_identity_checks", []).append({
            **observation, "role": role, "card_id": card_id,
            "contact": contact, "accepted": accepted,
        })

    def save(self, diagnostic: dict, outcome: str, error: str | None) -> dict | None:
        if not self.active() or "probe_sources" not in diagnostic:
            return None
        position = diagnostic["position"]
        card_id = position.get("observed_detail_card_id")
        p_item = position.get("kind") == "p_item"
        if outcome in {"cancelled", "superseded"} or (not p_item and card_id in self.RETIRED_TARGETS):
            return None
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
        if not re.fullmatch(r"[0-9a-f]{32}", str(diagnostic.get("transaction_id", ""))):
            return None
        folder = self.root.resolve() / "reader-failures"
        if folder.exists():
            if not stat.S_ISDIR(_plain_stat(folder).st_mode):
                return None
        records = self._records(folder)
        max_samples = min(self.MAX_SAMPLES, int(self.settings.get("max_samples", self.MAX_SAMPLES)))
        max_bytes = min(self.MAX_BYTES, int(self.settings.get("max_bytes", self.MAX_BYTES)))
        if max_samples <= 0 or max_bytes <= 0:
            return None
        if shutil.disk_usage(self.root).free < max_bytes + 256 * 1024 * 1024:
            return None
        import cv2

        images = []
        for index, image in enumerate(diagnostic["probe_sources"]):
            images.append((f"source-{index + 1}", image, None))
        identity_checks = diagnostic.get("probe_identity_checks", ())
        for check in identity_checks:
            if not any(image is check["image"] for _, image, _ in images):
                images.append((check["role"], check["image"], check["ocr"]))
        confirmed = diagnostic.get("confirmed_open")
        if identity_checks and confirmed is not None and not any(image is confirmed[1] for _, image, _ in images):
            images.append(("title-confirmed-open", confirmed[1], None))
        observations = [frame for frame in diagnostic["probe_frames"]
                        if not any(image is frame["image"] for _, image, _ in images)]
        if identity_checks:
            remaining_slots = max(0, 10 - len(images))
            if len(observations) > remaining_slots:
                recent_count = min(4, remaining_slots)
                observations = (observations[:remaining_slots - recent_count]
                                + (observations[-recent_count:] if recent_count else []))
        for frame in observations:
            images.append(("observed-" + frame["phase"], frame["image"], frame["ocr"]))
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
            "detail_id_checks": [
                {"file": f"{index:02d}.png", "role": check["role"],
                 "card_id": check["card_id"], "contact": check["contact"],
                 "accepted": check["accepted"], "reco_id": check.get("reco_id"),
                 "seconds": check.get("seconds")}
                for index, (_, img, _) in enumerate(images[:10])
                for check in identity_checks if check["image"] is img
            ],
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
            "reader_probe_revision": "arena-feedback-20260922-candidate-pairs-v3",
            "build": self.settings.get("build"),
            "truth_status": "unreviewed_observation_not_training_label",
            "extra_screenshots": 0, "extra_ocr": 0, "extra_clicks": 0,
        }
        metadata = json.dumps(evidence, ensure_ascii=False, indent=2).encode("utf-8")
        size = sum(map(len, encoded_frames)) + len(metadata)
        if size > max_bytes:
            return None
        victims = self._evictions(records, category, key, size, max_samples, max_bytes)
        if victims is None:
            return None
        destination = folder / ("probe-" + diagnostic["transaction_id"])
        if destination.exists():
            return None
        destination.mkdir(parents=True, exist_ok=False)
        completed = False
        try:
            # Write metadata last; only complete new samples can replace old
            # samples. Avoid renaming newly written directories on Windows.
            for frame, data in zip(saved, encoded_frames, strict=True):
                (destination / frame["file"]).write_bytes(data)
            (destination / "evidence.json").write_bytes(metadata)
            for victim in victims:
                self._remove_group(victim, folder)
            completed = True
        except OSError:
            return None
        finally:
            if not completed and destination.exists():
                self._remove_group(destination, folder)
        return {"event": "arena_recognition_probe_saved", "folder": str(destination),
                "card_id": card_id, "outcome": outcome, "frames": len(saved)}
