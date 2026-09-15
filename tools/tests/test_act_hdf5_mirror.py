import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.act_hdf5_mirror import mirror_act_episode_hdf5


class TestActHdf5Mirror(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="robostd_hdf5_mirror_"))

    def tearDown(self):
        if self.tmpdir.exists():
            shutil.rmtree(self.tmpdir)

    def _make_episode(self, path: Path):
        with h5py.File(path, "w") as f:
            f.attrs["sim"] = False

            T = 3
            vec = np.arange(T * 14, dtype=np.float32).reshape(T, 14)
            f.create_dataset("action", data=vec)

            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=vec + 100)
            obs.create_dataset("qvel", data=vec + 200)

            imgs = obs.create_group("images")
            left = np.zeros((T, 2, 3, 3), dtype=np.uint8)
            right = np.zeros((T, 2, 3, 3), dtype=np.uint8)
            for t in range(T):
                left[t, :, :, :] = np.array(
                    [
                        [[1 + t, 0, 0], [2 + t, 0, 0], [3 + t, 0, 0]],
                        [[4 + t, 0, 0], [5 + t, 0, 0], [6 + t, 0, 0]],
                    ],
                    dtype=np.uint8,
                )
                right[t, :, :, :] = np.array(
                    [
                        [[10 + t, 0, 0], [20 + t, 0, 0], [30 + t, 0, 0]],
                        [[40 + t, 0, 0], [50 + t, 0, 0], [60 + t, 0, 0]],
                    ],
                    dtype=np.uint8,
                )
            imgs.create_dataset("left_wrist", data=left)
            imgs.create_dataset("right_wrist", data=right)

            f.create_dataset("is_pad", data=np.array([False, False, True], dtype=bool))

    def test_mirror(self):
        inp = self.tmpdir / "episode_000000.hdf5"
        out = self.tmpdir / "episode_000000_mir.hdf5"
        self._make_episode(inp)

        mirror_act_episode_hdf5(inp, out, joint_mode="aloha14", swap_cameras=True, frame_batch=2)

        with h5py.File(inp, "r") as f_in, h5py.File(out, "r") as f_out:
            self.assertIn("action", f_out)
            self.assertIn("observations", f_out)
            self.assertIn("is_pad", f_out)
            np.testing.assert_array_equal(f_out["is_pad"][...], f_in["is_pad"][...])

            a_in = f_in["action"][...]
            a_out = f_out["action"][...]
            within = np.array([-1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
            expected = np.concatenate([a_in[:, 7:14] * within, a_in[:, 0:7] * within], axis=1)
            np.testing.assert_allclose(a_out, expected)

            imgs_in = f_in["observations/images"]
            imgs_out = f_out["observations/images"]
            self.assertIn("left_wrist", imgs_out)
            self.assertIn("right_wrist", imgs_out)

            left_from_right = imgs_in["right_wrist"][...][:, :, ::-1, :]
            right_from_left = imgs_in["left_wrist"][...][:, :, ::-1, :]
            np.testing.assert_array_equal(imgs_out["left_wrist"][...], left_from_right)
            np.testing.assert_array_equal(imgs_out["right_wrist"][...], right_from_left)

    def test_left_wrist_equivalent_left(self):
        inp = self.tmpdir / "episode_000001.hdf5"
        out = self.tmpdir / "episode_000001_mir.hdf5"

        with h5py.File(inp, "w") as f:
            f.attrs["sim"] = False
            vec = np.arange(2 * 14, dtype=np.float32).reshape(2, 14)
            f.create_dataset("action", data=vec)
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=vec)
            obs.create_dataset("qvel", data=vec)
            imgs = obs.create_group("images")

            left_wrist = np.zeros((2, 2, 3, 3), dtype=np.uint8)
            right = np.zeros((2, 2, 3, 3), dtype=np.uint8)
            left_wrist[:, :, :, 0] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
            right[:, :, :, 0] = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
            imgs.create_dataset("left_wrist", data=left_wrist)
            imgs.create_dataset("right", data=right)

        mirror_act_episode_hdf5(inp, out, joint_mode="aloha14", swap_cameras=True, frame_batch=8)

        with h5py.File(inp, "r") as f_in, h5py.File(out, "r") as f_out:
            imgs_in = f_in["observations/images"]
            imgs_out = f_out["observations/images"]

            self.assertIn("left_wrist", imgs_out)
            self.assertIn("right", imgs_out)
            self.assertNotIn("left", imgs_out)
            self.assertNotIn("right_wrist", imgs_out)

            expected_left_wrist = imgs_in["right"][...][:, :, ::-1, :]
            expected_right = imgs_in["left_wrist"][...][:, :, ::-1, :]
            np.testing.assert_array_equal(imgs_out["left_wrist"][...], expected_left_wrist)
            np.testing.assert_array_equal(imgs_out["right"][...], expected_right)
