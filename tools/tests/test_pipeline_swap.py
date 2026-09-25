from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.pipeline_core import Pipeline


def _create_video(path: Path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for _ in range(5):
        writer.write(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    writer.release()


def test_pipeline_swap(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    (input_dir / "videos/chunk-000/cam_left").mkdir(parents=True)
    (input_dir / "videos/chunk-000/cam_right").mkdir(parents=True)
    (input_dir / "data/chunk-000").mkdir(parents=True)
    (input_dir / "meta").mkdir(parents=True)
    _create_video(input_dir / "videos/chunk-000/cam_left/episode_000000.mp4")
    _create_video(input_dir / "videos/chunk-000/cam_right/episode_000000.mp4")
    pd.DataFrame({"frame": range(5), "data": range(5)}).to_parquet(
        input_dir / "data/chunk-000/episode_000000.parquet"
    )
    (input_dir / "meta/info.json").write_text(
        '{"features": {"observation.state": {"shape": [1], "names": ["state"]}}}',
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text("joint_mirror_rules: {test_rule: []}", encoding="utf-8")

    success = Pipeline(str(config), "test_rule").process_dataset(
        str(input_dir), str(output_dir), [{"name": "mirror"}]
    )
    assert success
    assert (output_dir / "videos/chunk-000/cam_right/episode_000000.mp4").exists()
    assert (output_dir / "videos/chunk-000/cam_left/episode_000000.mp4").exists()
