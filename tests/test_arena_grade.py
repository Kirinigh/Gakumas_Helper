from __future__ import annotations

import unittest

import numpy as np

from agent.arena_winrate.grade import (
    MemberSlotState,
    stage_member_cap,
    classify_member_slot,
    fixed_member_slot_boxes,
    team_stage_total_anchors,
)


class ArenaGradeTests(unittest.TestCase):
    def test_grade_stage_caps(self) -> None:
        self.assertEqual([stage_member_cap(1, stage) for stage in (1, 2, 3)], [1, 1, 1])
        self.assertEqual([stage_member_cap(2, stage) for stage in (1, 2, 3)], [2, 1, 1])
        for grade in (3, 4, 5, 6):
            self.assertEqual([stage_member_cap(grade, stage) for stage in (1, 2, 3)], [2, 2, 2])
        self.assertEqual([stage_member_cap(7, stage) for stage in (1, 2, 3)], [3, 3, 3])

    def test_lower_grades_use_enabled_prefix_of_fixed_three_column_layout(self) -> None:
        self.assertEqual(fixed_member_slot_boxes(720, 555, 116, 1), ((322, 555, 90, 116),))
        self.assertEqual(
            fixed_member_slot_boxes(720, 555, 116, 2),
            ((322, 555, 90, 116), (418, 555, 90, 116)),
        )
        self.assertEqual(
            fixed_member_slot_boxes(720, 555, 116, 3),
            ((322, 555, 90, 116), (418, 555, 90, 116), (514, 555, 90, 116)),
        )

    def test_opponent_comparison_uses_right_hand_member_group(self) -> None:
        self.assertEqual(
            fixed_member_slot_boxes(
                720,
                555,
                116,
                3,
                group_center_ratio=0.7361111111,
            ),
            ((389, 555, 90, 116), (485, 555, 90, 116), (581, 555, 90, 116)),
        )

    def test_opponent_comparison_selects_right_total_from_each_stage_pair(self) -> None:
        totals = (
            (64, 520, 68, 29),
            (408, 521, 68, 30),
            (407, 692, 69, 32),
            (64, 695, 68, 29),
            (67, 871, 64, 25),
            (410, 871, 65, 25),
        )
        self.assertEqual(
            team_stage_total_anchors(totals, own_team=False),
            ((408, 521, 68, 30), (407, 692, 69, 32), (410, 871, 65, 25)),
        )

    def test_opponent_comparison_rejects_unpaired_totals(self) -> None:
        with self.assertRaisesRegex(ValueError, "six stage totals"):
            team_stage_total_anchors(((64, 520, 68, 29),), own_team=False)

    def test_explicit_black_slot_is_empty(self) -> None:
        crop = np.zeros((116, 90, 3), dtype=np.uint8)
        crop[:5, :, :] = 180
        crop[-5:, :, :] = 180
        crop[:, :5, :] = 180
        crop[:, -5:, :] = 180
        state, metrics = classify_member_slot(crop)
        self.assertIs(state, MemberSlotState.EMPTY)
        self.assertGreaterEqual(metrics.inner_dark_pixel_ratio, 0.75)

    def test_dim_low_texture_placeholder_is_empty(self) -> None:
        crop = np.full((116, 90, 3), 80, dtype=np.uint8)
        crop[:, 45:, :] = 105
        state, metrics = classify_member_slot(crop)
        self.assertIs(state, MemberSlotState.EMPTY)
        self.assertLessEqual(metrics.inner_gray_mean, 100)

    def test_textured_portrait_is_occupied(self) -> None:
        grid_y, grid_x = np.indices((116, 90))
        values = ((grid_x * 17 + grid_y * 23) % 256).astype(np.uint8)
        crop = np.repeat(values[:, :, None], 3, axis=2)
        state, _ = classify_member_slot(crop)
        self.assertIs(state, MemberSlotState.OCCUPIED)

    def test_weak_non_black_slot_is_ambiguous(self) -> None:
        crop = np.full((116, 90, 3), 130, dtype=np.uint8)
        state, _ = classify_member_slot(crop)
        self.assertIs(state, MemberSlotState.AMBIGUOUS)


if __name__ == "__main__":
    unittest.main()
