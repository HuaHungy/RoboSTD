import os
import shutil
import logging
from pathlib import Path
from src.pipeline_core import Pipeline

# Setup logging
logging.basicConfig(level=logging.INFO)

def create_dummy_data(root_dir):
    root = Path(root_dir)
    if root.exists():
        shutil.rmtree(root)
    
    # Create structure
    (root / "videos/chunk-000/cam_left").mkdir(parents=True)
    (root / "videos/chunk-000/cam_right").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    
    # Create dummy files
    # Create a small valid mp4 file using ffmpeg if possible, or just an empty file if pipeline doesn't strictly validate content immediately (but it does validate with cv2)
    # So we need a valid mp4.
    
    # Create a dummy valid mp4
    import cv2
    import numpy as np
    
    def create_video(path):
        height, width = 64, 64
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
        for _ in range(5):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
    create_video(root / "videos/chunk-000/cam_left/episode_000000.mp4")
    create_video(root / "videos/chunk-000/cam_right/episode_000000.mp4")
    
    # Create dummy parquet
    import pandas as pd
    df = pd.DataFrame({'frame': range(5), 'data': range(5)})
    df.to_parquet(root / "data/chunk-000/episode_000000.parquet")
    
    # Create dummy meta
    with open(root / "meta/info.json", 'w') as f:
        f.write('{"features": {"observation.state": {"shape": [1], "names": ["state"]}}}')

def test_pipeline_swap():
    input_dir = "temp_test_input"
    output_dir = "temp_test_output"
    
    create_dummy_data(input_dir)
    
    # Initialize pipeline (mock config path)
    # We need a dummy config file or mock the init
    # But Pipeline expects a config path for JointProcessor
    
    # Create a dummy config
    config_path = "temp_config.yaml"
    with open(config_path, 'w') as f:
        f.write("rules: {test_rule: {}}")
        
    pipeline = Pipeline(config_path, "test_rule")
    
    # Run pipeline with mirror operation
    operations = [{'name': 'mirror'}]
    
    print("Running pipeline...")
    success = pipeline.process_dataset(input_dir, output_dir, operations)
    
    if not success:
        print("Pipeline failed!")
        return
    
    # Verify output structure
    # Output dir will have a timestamp subfolder
    out_path = Path(output_dir)
    subdirs = [d for d in out_path.iterdir() if d.is_dir()]
    if not subdirs:
        print("No output directory created!")
        return
        
    latest_output = subdirs[0]
    print(f"Output directory: {latest_output}")
    
    # Check if files exist in swapped locations
    # Original left -> should be in right
    expected_right = latest_output / "videos/chunk-000/cam_right/episode_000000.mp4"
    # Original right -> should be in left
    expected_left = latest_output / "videos/chunk-000/cam_left/episode_000000.mp4"
    
    if expected_right.exists():
        print("SUCCESS: Original left video found in 'cam_right' folder.")
    else:
        print(f"FAILURE: Original left video NOT found in 'cam_right' folder. Checked: {expected_right}")
        
    if expected_left.exists():
        print("SUCCESS: Original right video found in 'cam_left' folder.")
    else:
        print(f"FAILURE: Original right video NOT found in 'cam_left' folder. Checked: {expected_left}")

    # Clean up
    # shutil.rmtree(input_dir)
    # shutil.rmtree(output_dir)
    # os.remove(config_path)

if __name__ == "__main__":
    test_pipeline_swap()
