import unittest

import numpy as np

from src.robostd_stage2 import (
    CoordinationConstraints,
    ManipulationUnit,
    decompose_units,
    parse_constraints_dict,
    reconstruct_pseudo_bimanual,
    schedule_units,
)


class TestRoboSTDStage2(unittest.TestCase):
    def test_strict_constraints_reject_missing_units_and_cycles(self):
        with self.assertRaisesRegex(ValueError, "missing units"):
            parse_constraints_dict(
                {"rho": {"0": "L"}, "e_pre": [], "e_conf": []},
                unit_ids=[0, 1],
            )

    def test_structured_output_wire_format_is_normalized(self):
        constraints = parse_constraints_dict(
            {
                "rho": [{"unit_id": 0, "arm": "L"}, {"unit_id": 1, "arm": "R"}],
                "e_pre": [{"before": 0, "after": 1}],
                "e_conf": [{"a": 0, "b": 1}],
            },
            unit_ids=[0, 1],
        )
        self.assertEqual(constraints.rho, {0: "L", 1: "R"})
        self.assertEqual(constraints.e_pre, [(0, 1)])
        self.assertEqual(constraints.e_conf, [(0, 1)])
        with self.assertRaisesRegex(ValueError, "cycle"):
            parse_constraints_dict(
                {
                    "rho": {"0": "L", "1": "R"},
                    "e_pre": [[0, 1], [1, 0]],
                    "e_conf": [],
                },
                unit_ids=[0, 1],
            )

    def test_precedence_reorders_and_conflict_serializes(self):
        units = [
            ManipulationUnit(0, 0, 2, "place"),
            ManipulationUnit(1, 2, 5, "grasp"),
            ManipulationUnit(2, 5, 7, "inspect"),
        ]
        constraints = CoordinationConstraints(
            rho={0: "L", 1: "R", 2: "L"},
            e_pre=[(1, 0)],
            e_conf=[(0, 2)],
        )
        scheduled = {item.unit.id: item for item in schedule_units(units, constraints, "R")}
        self.assertGreaterEqual(scheduled[0].start, scheduled[1].end)
        self.assertGreaterEqual(scheduled[2].start, scheduled[0].end)

    def test_non_conflicting_opposite_arms_overlap(self):
        units = [ManipulationUnit(0, 0, 3, "left"), ManipulationUnit(1, 3, 5, "right")]
        constraints = CoordinationConstraints(rho={0: "L", 1: "R"}, e_pre=[], e_conf=[])
        scheduled = schedule_units(units, constraints, "R")
        self.assertEqual([(item.start, item.end) for item in scheduled], [(0, 3), (0, 2)])

    def test_left_source_uses_original_left_and_mirror_right(self):
        # Four dimensions means two values per arm. Values make each source obvious.
        state_orig = np.array([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=float)
        state_mir = np.array([[110, 111, 112, 113], [120, 121, 122, 123]], dtype=float)
        action_orig = state_orig + 1000
        action_mir = state_mir + 1000
        units = [ManipulationUnit(0, 0, 1, "left skill"), ManipulationUnit(1, 1, 2, "right skill")]
        constraints = CoordinationConstraints(
            rho={0: "L", 1: "R"}, e_pre=[(0, 1)], e_conf=[]
        )
        result = reconstruct_pseudo_bimanual(
            state_orig,
            state_mir,
            action_orig,
            action_mir,
            constraints,
            units,
            source_arm="L",
            action_mode="position",
        )
        np.testing.assert_array_equal(result.state[0, :2], state_orig[0, :2])
        np.testing.assert_array_equal(result.state[1, 2:], state_mir[1, 2:])
        self.assertEqual(result.plan[0]["skill_source"], "original")
        self.assertEqual(result.plan[1]["skill_source"], "mirror")

    def test_position_hold_repeats_current_configuration_and_adds_annotations(self):
        base = np.arange(24, dtype=float).reshape(6, 4)
        units = decompose_units(6, boundaries=[0, 3, 6], unit_names=["pick", "place"])
        constraints = CoordinationConstraints(
            rho={0: "L", 1: "R"}, e_pre=[(0, 1)], e_conf=[]
        )
        result = reconstruct_pseudo_bimanual(
            base,
            base + 100,
            base + 1000,
            base + 1100,
            constraints,
            units,
            source_arm="L",
            action_mode="position",
        )
        # R is inactive in frame 0, so its absolute-position action holds its state.
        np.testing.assert_array_equal(result.action[0, 2:], result.state[0, 2:])
        self.assertIn("L arm: execute pick", result.frame_metadata[0]["stage_annotation"])
        self.assertEqual(result.frame_metadata[3]["active_unit_right"], 1)


if __name__ == "__main__":
    unittest.main()
