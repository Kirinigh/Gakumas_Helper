from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.task275_item_coverage import (
    P_ITEMS,
    P_ITEM_REFERENCE_ROOT,
    PItemRecognitionAssetAuditError,
    build_report,
    build_recognition_asset_report,
)
from agent.p_item_recognition.reference import PItemReferenceError


class Task275ItemAssetAuditTests(unittest.TestCase):
    def test_tracked_jjc_gallery_is_only_an_offline_compatibility_asset(self) -> None:
        report = build_report()
        asset = report["recognition_asset_compatibility"]

        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(
            report["source_sha256"],
            "CDE85AE7CD857207E73044F9E1063DBACD04399E00CAF1CD91C4A6A32F9714C5",
        )
        self.assertEqual(
            asset["reuse_status"],
            "OFFLINE_BUSINESS_ID_COMPATIBILITY_ONLY",
        )
        self.assertFalse(asset["runtime_ready"])
        self.assertFalse(asset["non_produce_domain_exact_match"])
        self.assertEqual(asset["gallery_business_id_count"], 456)
        self.assertEqual(asset["non_produce_business_id_count"], 420)
        self.assertEqual(asset["missing_non_produce_item_ids"], [])
        self.assertEqual(len(asset["unknown_gallery_item_ids"]), 36)
        self.assertEqual(asset["stage_business_id_count"], 368)
        self.assertEqual(asset["recognition_asset_stage_business_id_count"], 343)
        self.assertIn(
            "nia_pro_live_item_roi_not_calibrated",
            asset["runtime_blockers"],
        )
        self.assertIn(
            "normal_plus_real_domain_calibration_incomplete",
            asset["runtime_blockers"],
        )

        plans = {plan["plan_type"]: plan for plan in asset["plans"]}
        # The published gallery extension is ahead of the offline semantic CSV.
        expected = {
            "SENSE": (74, 70, ["3", "4", "274", "275"]),
            "LOGIC": (69, 65, ["5", "6", "276", "277"]),
            "ANOMALY": (55, 51, ["166", "167", "278", "279"]),
        }
        for plan_type, (supported, covered, missing) in expected.items():
            with self.subTest(plan_type=plan_type):
                plan = plans[plan_type]
                self.assertEqual(plan["semantic_supported_rows"], supported)
                self.assertEqual(plan["recognition_asset_supported_rows"], covered)
                self.assertEqual(plan["semantic_supported_missing_from_asset_ids"], missing)

    def test_gallery_hash_mismatch_fails_before_business_id_audit(self) -> None:
        with patch(
            "tools.task275_item_coverage._sha256_file",
            return_value="0" * 64,
        ):
            with self.assertRaisesRegex(
                PItemRecognitionAssetAuditError,
                "SHA-256 mismatch",
            ):
                build_recognition_asset_report()

    def test_manifest_domain_mismatch_is_rejected(self) -> None:
        manifest = json.loads((P_ITEM_REFERENCE_ROOT / "manifest.json").read_text(encoding="utf-8"))
        manifest["source"]["scope"] = "contest only"
        with patch("tools.task275_item_coverage.json.loads", return_value=manifest):
            with self.assertRaisesRegex(
                PItemRecognitionAssetAuditError,
                "approved non-produce scope",
            ):
                build_recognition_asset_report()

    def test_manifest_revision_mismatch_is_rejected(self) -> None:
        manifest = json.loads((P_ITEM_REFERENCE_ROOT / "manifest.json").read_text(encoding="utf-8"))
        manifest["source"]["revision"] = "other-source+gakumas-tools@3e9e8ebdc929dedd32cdd8d8911e5d8a905f76b6"
        with patch("tools.task275_item_coverage.json.loads", return_value=manifest):
            with self.assertRaisesRegex(
                PItemRecognitionAssetAuditError,
                "source revision does not match",
            ):
                build_recognition_asset_report()

    def test_semantic_catalog_drift_fails_closed(self) -> None:
        with patch(
            "tools.task275_item_coverage._normalised_lf_sha256",
            return_value="0" * 64,
        ):
            with self.assertRaisesRegex(
                PItemRecognitionAssetAuditError,
                "semantic catalog LF-normalised SHA-256 mismatch",
            ):
                build_recognition_asset_report()

    def test_gallery_path_escape_is_rejected(self) -> None:
        manifest = json.loads((P_ITEM_REFERENCE_ROOT / "manifest.json").read_text(encoding="utf-8"))
        manifest["reference_gallery"]["path"] = "../outside.npz"
        with patch("tools.task275_item_coverage.json.loads", return_value=manifest):
            with self.assertRaisesRegex(
                PItemRecognitionAssetAuditError,
                "path must stay inside",
            ):
                build_recognition_asset_report()

    def test_gallery_declared_count_mismatch_is_rejected(self) -> None:
        manifest = json.loads((P_ITEM_REFERENCE_ROOT / "manifest.json").read_text(encoding="utf-8"))
        manifest["reference_gallery"]["business_id_count"] = 419
        with patch("tools.task275_item_coverage.json.loads", return_value=manifest):
            with self.assertRaisesRegex(
                PItemRecognitionAssetAuditError,
                "manifest count does not match the gallery ID set",
            ):
                build_recognition_asset_report()

    def test_gallery_loader_failure_is_rejected(self) -> None:
        with patch(
            "tools.task275_item_coverage.PItemRenderedReferenceGallery.load",
            side_effect=PItemReferenceError("synthetic NPZ failure"),
        ):
            with self.assertRaisesRegex(
                PItemRecognitionAssetAuditError,
                "synthetic NPZ failure",
            ):
                build_recognition_asset_report()

    def test_valid_but_incomplete_gallery_remains_runtime_blocked(self) -> None:
        with np.load(
            P_ITEM_REFERENCE_ROOT / "p_item_rendered_reference_gallery.npz",
            allow_pickle=False,
        ) as arrays:
            ids = tuple(int(item_id) for item_id in arrays["p_item_ids"] if item_id != 7)
        manifest = json.loads((P_ITEM_REFERENCE_ROOT / "manifest.json").read_text(encoding="utf-8"))
        manifest["reference_gallery"]["business_id_count"] = len(ids)
        with (
            patch("tools.task275_item_coverage.json.loads", return_value=manifest),
            patch(
                "tools.task275_item_coverage.PItemRenderedReferenceGallery.load",
                return_value=SimpleNamespace(p_item_ids=ids),
            ),
        ):
            report = build_recognition_asset_report(p_items_path=P_ITEMS)

        self.assertFalse(report["runtime_ready"])
        self.assertFalse(report["non_produce_domain_exact_match"])
        self.assertEqual(report["missing_non_produce_item_ids"], ["7"])
        sense = next(plan for plan in report["plans"] if plan["plan_type"] == "SENSE")
        self.assertIn("7", sense["semantic_supported_missing_from_asset_ids"])
        self.assertIn(
            "reference_and_semantic_business_id_sets_disagree",
            report["runtime_blockers"],
        )


if __name__ == "__main__":
    unittest.main()
