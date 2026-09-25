import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.geometry_mirror import (
    derive_joint_signs_from_axes,
    derive_joint_signs_from_urdf,
    reflection_matrix,
)


class TestGeometryMirror(unittest.TestCase):
    def test_reflection_is_symmetric_involutory_and_improper(self):
        reflection = reflection_matrix((0, 1, 0))
        np.testing.assert_allclose(reflection, reflection.T)
        np.testing.assert_allclose(reflection @ reflection, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(reflection), -1.0)

    def test_axis_rule_matches_equation_seven(self):
        signs = derive_joint_signs_from_axes(
            source_axes=[(0, 0, 1), (1, 0, 0)],
            target_axes=[(0, 0, -1), (1, 0, 0)],
            normal=(0, 1, 0),
        )
        self.assertEqual(signs, [1, -1])

    def test_urdf_axes_are_resolved_in_common_frame(self):
        urdf = """<robot name="symmetric">
          <link name="base"/><link name="left"/><link name="right"/>
          <joint name="left_joint" type="revolute">
            <parent link="base"/><child link="left"/><origin xyz="0 1 0" rpy="0 0 0"/>
            <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
          </joint>
          <joint name="right_joint" type="revolute">
            <parent link="base"/><child link="right"/><origin xyz="0 -1 0" rpy="0 0 0"/>
            <axis xyz="0 0 -1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
          </joint>
        </robot>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.urdf"
            path.write_text(urdf, encoding="utf-8")
            self.assertEqual(
                derive_joint_signs_from_urdf(path, ["left_joint"], ["right_joint"]),
                [1],
            )


if __name__ == "__main__":
    unittest.main()
