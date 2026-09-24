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
        identity = {"version": build.get("derived_version"),
                    "release_source": build.get("source", {}).get("revision")}
        framework = build.get("framework")
        if isinstance(framework, dict) and isinstance(framework.get("version"), str):
            identity["framework"] = framework["version"]
        patches = build.get("local_patches")
        if isinstance(patches, list) and patches and isinstance(patches[-1], dict):
            revision = patches[-1].get("source_revision")
            if isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                identity["patch_source"] = revision
        return {"enabled": True, "campaign": CAMPAIGN,
                "starts_at": "2026-09-20T00:00:00+08:00", "expires_at": EXPIRES_AT,
                "max_samples": 20, "max_bytes": 128 * 1024 * 1024,
                "build": identity}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


class RecognitionProbe:
    TARGETS = frozenset({31, 297, 403, 553})
    RETIRED_TARGETS = frozenset({299, 389, 752})
    MAX_SAMPLES = 20
    MAX_BYTES = 128 * 1024 * 1024
    SOURCE_PNG_CACHE_BYTES = 24 * 1024 * 1024

    def __init__(self, root: Path):
        self.root = root
        self.settings: dict[str, Any] = release_probe_settings()
        release_build = self.settings.get("build")
        self.disabled = False
        self._source_png_member: str | None = None
        self._source_png_cache: list[tuple[Any, Any, bytes, int]] = []
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

    def _records(self, folder: Path, *, incomplete: list[Path] | None = None) -> list[dict]:
        records = []
        for group in folder.glob("probe-*"):
            if not re.fullmatch(r"probe-[0-9a-f]{32}", group.name):
                continue
            try:
                files = self._group_files(group, folder)
            except OSError:
                # Unknown contents and links remain outside probe ownership.
                continue
            try:
                metadata = group / "evidence.json"
                if metadata.stat().st_size > 1024 * 1024:
                    raise ValueError("probe metadata is oversized")
                value = json.loads(metadata.read_text(encoding="utf-8"))
                if value.get("event") != "arena_recognition_probe_sample":
                    continue
                recorded = datetime.fromisoformat(value["recorded_at"])
                if recorded.tzinfo is None:
                    raise ValueError("probe timestamp has no timezone")
                sizes = {path.name: path.stat().st_size for path in files}
                frames = [frame["file"] for frame in value["frames"]]
                if (not frames or len(frames) != len(set(frames))
                        or any(not isinstance(name, str) or not _FRAME_NAME.fullmatch(name)
                               or sizes.get(name, 0) <= 0 for name in frames)):
                    raise ValueError("probe frame set is incomplete")
                replaced = value.get("replaced_groups", [])
                if (not isinstance(replaced, list)
                        or any(not isinstance(name, str) or not re.fullmatch(r"probe-[0-9a-f]{32}", name)
                               for name in replaced)):
                    raise ValueError("probe replacement groups are invalid")
                records.append({"path": group, "value": value, "at": recorded.timestamp(),
                                "bytes": sum(sizes.values())})
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                if incomplete is not None:
                    incomplete.append(group)
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

    def _source_png(self, image: Any) -> bytes:
        """Reuse only owned uint8 source arrays whose pixels still match."""
        import cv2
        import numpy as np

        eligible = (
            type(image) is np.ndarray and image.dtype == np.uint8
            and image.flags.owndata and image.base is None and image.flags.c_contiguous
            and (image.ndim == 2 or image.ndim == 3 and image.shape[2] in (1, 3, 4))
            and 2 * image.nbytes + 1024 <= self.SOURCE_PNG_CACHE_BYTES
        )
        # Shape changes invalidate retained arrays before accounting/reuse.
        self._source_png_cache[:] = [
            entry for entry in self._source_png_cache
            if entry[0].shape == entry[1].shape and entry[0].dtype == entry[1].dtype
        ]
        for index, (original, snapshot, data, _size) in enumerate(self._source_png_cache):
            if original is image:
                if eligible and np.array_equal(image, snapshot):
                    return data
                self._source_png_cache.pop(index)
                break
        snapshot = image.copy() if eligible else image
        ok, encoded = cv2.imencode(".png", snapshot)
        if not ok:
            raise ValueError("probe PNG encoding failed")
        data = encoded.tobytes()
        if eligible:
            # Count both retained pixel arrays, encoded bytes and a conservative
            # allowance for their Python containers; views are never retained.
            size = 2 * image.nbytes + len(data) + 1024
            if size <= self.SOURCE_PNG_CACHE_BYTES:
                while self._source_png_cache and (
                    len(self._source_png_cache) >= 3
                    or sum(entry[3] for entry in self._source_png_cache) + size > self.SOURCE_PNG_CACHE_BYTES
                ):
                    self._source_png_cache.pop(0)
                self._source_png_cache.append((image, snapshot, data, size))
        return data

    def save(self, diagnostic: dict, outcome: str, error: str | None) -> dict | None:
        if not self.active() or "probe_sources" not in diagnostic:
            self._source_png_cache.clear()
            self._source_png_member = None
            return None
        position = diagnostic["position"]
        card_id = position.get("observed_detail_card_id")
        p_item = position.get("kind") == "p_item"
        cost_evidence = diagnostic.get("probe_cost_evidence") if not p_item else None
        member = diagnostic.get("member_read_id")
        cache_sources = (
            not p_item and len(diagnostic["probe_sources"]) == 3
            and isinstance(member, str) and re.fullmatch(r"[0-9a-f]{32}", member) is not None
            and outcome not in {"cancelled", "superseded"}
        )
        if not cache_sources or member != self._source_png_member:
            self._source_png_cache.clear()
            self._source_png_member = member if cache_sources else None
        if outcome in {"cancelled", "superseded"} or (not p_item and not cost_evidence and card_id in self.RETIRED_TARGETS):
            return None
        if p_item and outcome != "completed" and "p_item_source_restore_unproven" not in str(error):
            return None
        target = card_id if card_id in self.TARGETS else None
        # Non-target samples are controls, never automatically "unknown cards".
        category = ("p_item_return_control" if outcome == "completed" else "p_item_return_failed") if p_item else (str(target) if target else "control")
        if cost_evidence:
            category = "cost_evidence_inconclusive"
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
        incomplete = []
        records = self._records(folder, incomplete=incomplete)
        max_samples = min(self.MAX_SAMPLES, int(self.settings.get("max_samples", self.MAX_SAMPLES)))
        max_bytes = min(self.MAX_BYTES, int(self.settings.get("max_bytes", self.MAX_BYTES)))
        if max_samples <= 0 or max_bytes <= 0:
            return None
        # A failed rollback/cleanup may leave residue or one replacement overlap.
        # Do not add files, even after restart, until existing cleanup removes it.
        if incomplete or len(records) > max_samples or sum(r["bytes"] for r in records) > max_bytes:
            return None
        if any((folder / name).exists() for record in records
               for name in record["value"].get("replaced_groups", [])):
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
            if cache_sources and index < 3:
                data = self._source_png(image)
            else:
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
            "cost_evidence": cost_evidence,
            "reader_probe_revision": "arena-feedback-20260923-cost-evidence-v4",
            "build": self.settings.get("build"),
            "truth_status": "unreviewed_observation_not_training_label",
            "extra_screenshots": 0, "extra_ocr": 0, "extra_clicks": 0,
        }
        victims = []
        while True:
            # Persist replacement ownership before deletion, even below the
            # global limits. Added names can require another byte-budget victim.
            evidence["replaced_groups"] = [victim.name for victim in victims]
            metadata = json.dumps(evidence, ensure_ascii=False, indent=2).encode("utf-8")
            size = sum(map(len, encoded_frames)) + len(metadata)
            if size > max_bytes:
                return None
            selected = self._evictions(records, category, key, size, max_samples, max_bytes)
            if selected is None:
                return None
            if selected == victims:
                break
            victims = selected
        destination = folder / ("probe-" + diagnostic["transaction_id"])
        if destination.exists():
            return None
        try:
            # Bounded by the existing sample budget; no extra backup files.
            backups = {victim: {path.name: path.read_bytes()
                                for path in self._group_files(victim, folder)}
                       for victim in victims}
        except OSError:
            return None
        completed = False
        try:
            destination.mkdir(parents=True, exist_ok=False)
            # Write metadata last; only complete new samples can replace old
            # samples. Avoid renaming newly written directories on Windows.
            for frame, data in zip(saved, encoded_frames, strict=True):
                (destination / frame["file"]).write_bytes(data)
            (destination / "evidence.json").write_bytes(metadata)
            completed = True
            for victim in victims:
                self._remove_group(victim, folder)
        except OSError:
            if completed:
                try:
                    for victim, files in backups.items():
                        victim.mkdir(exist_ok=True)
                        self._group_files(victim, folder)
                        for name, data in files.items():
                            path = victim / name
                            if not path.exists():
                                path.write_bytes(data)
                except OSError:
                    # The complete replacement is now the only reliable copy.
                    # Retain it; the inventory guard pauses further saves.
                    return None
            try:
                if destination.exists():
                    self._remove_group(destination, folder)
            except OSError:
                pass  # The next inventory scan also blocks partial new writes.
            return None
        return {"event": "arena_recognition_probe_saved", "folder": str(destination),
                "card_id": card_id, "outcome": outcome, "frames": len(saved)}
