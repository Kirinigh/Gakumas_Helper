from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.arena_winrate import StageCatalogError, ContestStageCatalog
from agent.arena_winrate.stages import CATALOG_RELATIVE_PATH


def stage(stage_id: int, season: int, stage_number: int, *, preview: bool = False) -> dict[str, object]:
    return {
        "id": stage_id,
        "name": f"season {season} stage {stage_number}",
        "type": "contest",
        "preview": preview,
        "season": season,
        "stage": stage_number,
        "plan": "logic",
        "criteria": "0.45,0.35,0.2",
        "turnCounts": "6,4,2",
        "firstTurns": "1,0,0",
        "effects": "at:startOfTurn { motivation+=1 }",
        "linkTurnCounts": "",
    }


def rows() -> list[dict[str, object]]:
    return [
        *(stage(169 + offset, 49, offset + 1) for offset in range(3)),
        *(stage(172 + offset, 50, offset + 1, preview=True) for offset in range(3)),
        {**stage(999, 99, 1), "type": "event"},
    ]


class ContestStageCatalogTests(unittest.TestCase):
    def test_latest_uses_highest_catalog_season_and_preserves_preview(self) -> None:
        season = ContestStageCatalog.from_rows(rows()).resolve("latest")
        self.assertEqual(season.season, 50)
        self.assertEqual(season.stage_ids, (172, 173, 174))
        self.assertTrue(season.preview)

    def test_manual_season_maps_all_three_stage_ids(self) -> None:
        season = ContestStageCatalog.from_rows(rows()).resolve(49)
        self.assertEqual(season.stage_ids, (169, 170, 171))
        self.assertFalse(season.preview)

    def test_unknown_or_incomplete_season_fails_closed(self) -> None:
        catalog = ContestStageCatalog.from_rows(rows()[:-2])
        with self.assertRaisesRegex(StageCatalogError, "exactly stages 1, 2 and 3"):
            catalog.resolve("latest")
        with self.assertRaisesRegex(StageCatalogError, "not present"):
            catalog.resolve(51)

    def test_catalog_is_loaded_from_the_version_matched_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / CATALOG_RELATIVE_PATH
            catalog_path.parent.mkdir(parents=True)
            catalog_path.write_text(json.dumps(rows()), encoding="utf-8")
            season = ContestStageCatalog.from_bundle(directory).resolve(50)
        self.assertEqual(season.stage_ids, (172, 173, 174))


if __name__ == "__main__":
    unittest.main()
