import os
import shutil
import unittest
import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append("/home/huahungy/RoboSTD/RoboSTD_Unified")
from scripts.corobot_mirror_tool import process_directory

class TestMirrorTool(unittest.TestCase):
    def setUp(self):
        self.test_root = Path("test_mirror_data")
        if self.test_root.exists(): shutil.rmtree(self.test_root)
        self.test_root.mkdir()
        
        # Create structure
        self.chunk_dir = self.test_root / "chunk-000"
        self.chunk_dir.mkdir()
        
        self.high_dir = self.chunk_dir / "observation.images.cam_high_rgb"
        self.left_dir = self.chunk_dir / "observation.images.cam_left_wrist_rgb"
        self.right_dir = self.chunk_dir / "observation.images.cam_right_wrist_rgb"
        
        self.high_dir.mkdir()
        self.left_dir.mkdir()
        self.right_dir.mkdir()
        
        # Create dummy videos
        # High: Top-Left White (Mirrored -> Top-Right White)
        self.create_video(self.high_dir / "high.mp4", mark_pos=(25, 25))
        
        # Left: Bottom-Left White (Mirrored -> Bottom-Right White) -> Goes to Right Folder
        self.create_video(self.left_dir / "left.mp4", mark_pos=(75, 25))
        
        # Right: Bottom-Right White (Mirrored -> Bottom-Left White) -> Goes to Left Folder
        self.create_video(self.right_dir / "right.mp4", mark_pos=(75, 75))

    def tearDown(self):
        if self.test_root.exists(): shutil.rmtree(self.test_root)
        if Path("test_mirror_data_backups").exists(): shutil.rmtree("test_mirror_data_backups") # In case it creates outside

    def create_video(self, path, mark_pos):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, 30, (100, 100))
        for _ in range(10):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            # Mark a 10x10 square
            y, x = mark_pos
            frame[y-5:y+5, x-5:x+5] = 255
            out.write(frame)
        out.release()

    def check_video_mark(self, path, expected_pos):
        cap = cv2.VideoCapture(str(path))
        ret, frame = cap.read()
        cap.release()
        self.assertTrue(ret)
        
        y, x = expected_pos
        # Check if the expected position is white (or close to it due to compression)
        # Check center of the mark
        pixel = frame[y, x]
        # Allow some tolerance
        self.assertTrue(np.mean(pixel) > 100, f"Pixel at {y},{x} should be white but is {pixel}")

    def test_mirror_process(self):
        print("\nRunning Mirror Tool Test...")
        success = process_directory(str(self.test_root))
        self.assertTrue(success)
        
        # 1. Verify Backup
        backup_root = self.test_root.parent / "backups"
        self.assertTrue(backup_root.exists())
        backups = list(backup_root.glob("backup_*"))
        self.assertTrue(len(backups) > 0)
        
        # 2. Verify High (In-place mirror)
        # Original: (25, 25) [Top-Left] -> Mirrored: (25, 75) [Top-Right]
        self.check_video_mark(self.high_dir / "high.mp4", (25, 75))
        
        # 3. Verify Left/Right Swap & Mirror
        
        # Original Left Video: (75, 25) [Bottom-Left]
        # Mirrored: (75, 75) [Bottom-Right]
        # Destination: Right Folder
        # Check Right Folder for video named "left.mp4" (Name preserved? Yes, usually dataset has matching names like episode_0.mp4)
        # Wait, if I have left/episode_0.mp4 and right/episode_0.mp4.
        # After swap:
        # right/episode_0.mp4 should contain the mirrored content of ORIGINAL left/episode_0.mp4.
        
        # In my setup: left/left.mp4 and right/right.mp4.
        # Result:
        # left/right.mp4 (from original right)
        # right/left.mp4 (from original left)
        
        # Original Right Video: (75, 75) [Bottom-Right]
        # Mirrored: (75, 25) [Bottom-Left]
        # Destination: Left Folder
        
        self.assertTrue((self.left_dir / "right.mp4").exists())
        self.assertTrue((self.right_dir / "left.mp4").exists())
        
        # Check content of left/right.mp4 (Should be mirrored original right)
        # Original Right: (75, 75) -> Mirrored: (75, 25)
        self.check_video_mark(self.left_dir / "right.mp4", (75, 25))
        
        # Check content of right/left.mp4 (Should be mirrored original left)
        # Original Left: (75, 25) -> Mirrored: (75, 75)
        self.check_video_mark(self.right_dir / "left.mp4", (75, 75))

if __name__ == '__main__':
    unittest.main()
