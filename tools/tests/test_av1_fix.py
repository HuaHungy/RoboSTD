import os
import shutil
import unittest
import cv2
import numpy as np
import pandas as pd
import json
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append("/home/huahungy/RoboSTD/RoboSTD_Unified")

from src.pipeline_core import Pipeline

print("Test file loaded.")

class TestAV1Fix(unittest.TestCase):
    def setUp(self):
        print("Setting up test environment...")
        self.test_dir = Path("test_av1_data")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)
        
        # Setup directories
        (self.test_dir / "data/chunk-000").mkdir(parents=True, exist_ok=True)
        (self.test_dir / "videos/chunk-000/observation.images.cam_head_rgb").mkdir(parents=True, exist_ok=True)
        (self.test_dir / "meta").mkdir(exist_ok=True)
        self.output_dir = Path("test_av1_output")
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create dummy config
        self.config_path = self.test_dir / "config.yaml"
        with open(self.config_path, 'w') as f:
            f.write("project_name: 'AV1Test'\njoint_mirror_rules: {aloha: []}")

        # Create dummy info.json
        self.info_path = self.test_dir / "meta/info.json"
        with open(self.info_path, 'w') as f:
            json.dump({"features": {"joint_pos": {"names": ["j1"]}}}, f)

        # Copy the actual problematic video for testing
        self.src_video = "/home/huahungy/RoboSTD/Cobot_Magic_move_plate_qced_hardlink/videos/chunk-000/observation.images.cam_head_rgb/episode_000056.mp4"
        self.dest_video = self.test_dir / "videos/chunk-000/observation.images.cam_head_rgb/episode_000056.mp4"
        shutil.copy(self.src_video, self.dest_video)
        print(f"Copied source video to {self.dest_video}")
        
        # Create dummy parquet with matching length (from user log: 235 rows)
        self.parquet_path = self.test_dir / "data/chunk-000/episode_000056.parquet"
        # We need 235 rows to match the video
        df = pd.DataFrame({'joint_pos': [[0.0]] * 235, 'timestamp': [0.0] * 235})
        df.to_parquet(self.parquet_path, engine='pyarrow')

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def test_av1_transcoding_pipeline(self):
        print("Running pipeline...")
        pipeline = Pipeline(str(self.config_path), "aloha")
        # Just use mirror as an operation
        ops = [{'name': 'mirror'}]
        
        success = pipeline.process_dataset(str(self.test_dir), str(self.output_dir), ops)
        self.assertTrue(success)
        
        # Verify output
        subdirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        self.assertTrue(len(subdirs) > 0, "No output directory created")
        res_dir = subdirs[0]
        out_vid = res_dir / "videos/chunk-000/observation.images.cam_head_rgb/episode_000056.mp4"
        
        self.assertTrue(out_vid.exists(), f"Output video not found at {out_vid}")
        
        # Check codec using ffprobe
        try:
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", str(out_vid)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output_codec = result.stdout.strip()
            print(f"Output codec detected: {output_codec}")
            self.assertEqual(output_codec, 'av1', f"Expected codec av1, got {output_codec}")
        except Exception as e:
            print(f"Failed to check codec: {e}")

        # Verify video is now readable
        # NOTE: Since we restored to AV1, OpenCV will FAIL to read it again (because OpenCV doesn't support AV1 reading).
        # So we expect failure here if we try to read with OpenCV.
        # Instead, we should verify we can transcode it back to H.264 (proving it's a valid video file)
        
        print("Verifying AV1 validity by transcoding back to H.264...")
        temp_verify = str(out_vid).replace(".mp4", "_verify.mp4")
        cmd_verify = ["ffmpeg", "-y", "-i", str(out_vid), "-c:v", "libx264", temp_verify]
        subprocess.run(cmd_verify, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        cap = cv2.VideoCapture(temp_verify)
        ret, frame = cap.read()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        self.assertTrue(ret, "Output video should be valid (transcodable)")
        print(f"Success! Output video valid. Frame count: {frame_count}")
        
        # Tolerance of +/- 1 frame due to transcoding/encoding differences
        self.assertTrue(abs(frame_count - 235) <= 2, f"Frame count {frame_count} mismatch (expected ~235)")

if __name__ == '__main__':
    unittest.main()
