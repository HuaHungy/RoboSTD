import os
import shutil
import unittest
import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from src.pipeline_core import Pipeline
from src.joint_processor import JointProcessor

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        
        # Setup directories
        (self.test_dir / "data/chunk-000").mkdir(parents=True, exist_ok=True)
        (self.test_dir / "videos/chunk-000/observation.images.cam_left_wrist_rgb").mkdir(parents=True, exist_ok=True)
        (self.test_dir / "meta").mkdir(exist_ok=True)
        self.output_dir = Path("test_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create dummy config
        self.config_path = self.test_dir / "config.yaml"
        with open(self.config_path, 'w') as f:
            f.write("""
project_name: "TestProject"
joint_mirror_rules:
  test_rule:
    - source: joint_1
      target: joint_1
      scale: -1.0
""")

        # Create dummy info.json
        self.info_path = self.test_dir / "meta/info.json"
        with open(self.info_path, 'w') as f:
            json.dump({
                "features": {
                    "joint_pos": {
                        "names": ["joint_1", "joint_2"]
                    }
                }
            }, f)

        # Create dummy video
        self.video_path = self.test_dir / "videos/chunk-000/observation.images.cam_left_wrist_rgb/episode_000000.mp4"
        self.create_dummy_video(str(self.video_path))
        
        # Create dummy parquet
        self.parquet_path = self.test_dir / "data/chunk-000/episode_000000.parquet"
        df = pd.DataFrame({
            'joint_pos': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            'timestamp': [0.0, 0.1, 0.2]
        })
        # Parquet needs pyarrow
        try:
            df.to_parquet(self.parquet_path, engine='pyarrow')
        except ImportError:
            # Fallback if pyarrow not installed? Or skip test.
            pass

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def create_dummy_video(self, path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, 30, (100, 100))
        for _ in range(3): # 3 frames
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            # Add something to identify orientation
            frame[0:50, 0:50] = 255 # Top-Left white
            out.write(frame)
        out.release()

    def test_pipeline_mirror(self):
        if not self.parquet_path.exists():
            print("Skipping test due to missing pyarrow/parquet support")
            return

        pipeline = Pipeline(str(self.config_path), "test_rule")
        ops = [{'name': 'mirror'}]
        
        success = pipeline.process_dataset(str(self.test_dir), str(self.output_dir), ops)
        self.assertTrue(success)
        
        # Verify output exists
        # Find timestamp dir
        subdirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        self.assertEqual(len(subdirs), 1)
        res_dir = subdirs[0]
        
        out_vid = res_dir / "videos/chunk-000/observation.images.cam_left_wrist_rgb/episode_000000.mp4"
        out_pq = res_dir / "data/chunk-000/episode_000000.parquet"
        
        self.assertTrue(out_vid.exists())
        self.assertTrue(out_pq.exists())
        
        # Verify video content (flipped)
        cap = cv2.VideoCapture(str(out_vid))
        ret, frame = cap.read()
        cap.release()
        self.assertTrue(ret)
        # Original had white at Top-Left. Flipped should have white at Top-Right.
        # Top-Left: 0:50, 0:50. Top-Right: 0:50, 50:100.
        # Check pixel at 25, 75 (should be white 255)
        # Check pixel at 25, 25 (should be black 0)
        self.assertTrue(np.all(frame[25, 75] == 255) or np.all(frame[25, 75] > 200)) # Compression artifacts?
        self.assertTrue(np.all(frame[25, 25] == 0) or np.all(frame[25, 25] < 50))

        # Verify parquet content (negated)
        df = pd.read_parquet(out_pq)
        # Rule: joint_1 * -1. joint_2 unchanged (no rule).
        # Original: [1.0, 2.0]
        # Expected: [-1.0, 2.0]
        
        # df['joint_pos'][0] should be [-1.0, 2.0]
        row0 = df['joint_pos'].iloc[0]
        # Check if list or array
        if isinstance(row0, np.ndarray):
            row0 = row0.tolist()
            
        self.assertAlmostEqual(row0[0], -1.0)
        self.assertAlmostEqual(row0[1], 2.0)

if __name__ == '__main__':
    unittest.main()
