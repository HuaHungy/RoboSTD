import unittest

import numpy as np
import pandas as pd

from src.dataset_reconstruction import reconstruct_dataframe
from src.robostd_stage2 import (
    CoordinationConstraints,
    ManipulationUnit,
    reconstruct_pseudo_bimanual,
)


class TestDatasetReconstruction(unittest.TestCase):
    def test_dataframe_contains_aligned_language_and_provenance(self):
        vectors = [np.asarray(row, dtype=np.float32) for row in np.arange(16).reshape(4, 4)]
        original = pd.DataFrame(
            {
                "observation.state": vectors,
                "action": vectors,
                "frame_index": range(4),
                "timestamp": np.arange(4) / 10,
                "index": range(10, 14),
                "marker": ["o0", "o1", "o2", "o3"],
            }
        )
        mirrored = original.copy(deep=True)
        mirrored["observation.state"] = [row + 100 for row in vectors]
        mirrored["action"] = [row + 100 for row in vectors]
        mirrored["marker"] = ["m0", "m1", "m2", "m3"]
        units = [ManipulationUnit(0, 0, 2, "pick"), ManipulationUnit(1, 2, 4, "place")]
        constraints = CoordinationConstraints(
            rho={0: "L", 1: "R"}, e_pre=[(0, 1)], e_conf=[]
        )
        result = reconstruct_pseudo_bimanual(
            np.vstack(original["observation.state"]),
            np.vstack(mirrored["observation.state"]),
            np.vstack(original["action"]),
            np.vstack(mirrored["action"]),
            constraints,
            units,
            source_arm="R",
        )
        output = reconstruct_dataframe(original, mirrored, result, "move object", fps=20)
        self.assertEqual(output["frame_index"].tolist(), [0, 1, 2, 3])
        self.assertEqual(output["index"].tolist(), [0, 1, 2, 3])
        self.assertEqual(output["marker"].tolist(), ["m0", "m1", "o2", "o3"])
        self.assertTrue(output["robostd.language"].str.startswith("move object.").all())
        self.assertEqual(output["robostd.observation_source"].tolist(), ["mirror", "mirror", "original", "original"])


if __name__ == "__main__":
    unittest.main()
