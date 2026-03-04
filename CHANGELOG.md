# Changelog

## [Unreleased] - 2026-03-04

### Added
- **CoRobot Mirror Tool**: Added `scripts/corobot_mirror_tool.py`, a specialized standalone tool for mirroring CoRobot datasets. It supports:
    - Automatic backup and rollback.
    - In-place mirroring for "high" and "head" camera folders.
    - Mirroring and **swapping** content for "left" and "right" camera folders.
- **AV1 Codec Support**: The main pipeline now automatically detects AV1 videos. It temporarily converts them to H.264 for OpenCV processing and restores the original codec (or H.264) in the final output to ensure compatibility and playback.
- **Folder Swapping in Pipeline**: The main pipeline (`src/pipeline_core.py`) now includes logic to automatically swap "left" and "right" folder names in the output path when the `mirror` operation is applied, ensuring semantic correctness for stereo camera setups.
- **Robustness**: 
    - Added a `tools/tests/test_pipeline_swap.py` to verify folder swapping logic.
    - Decoupled video processing from joint data processing. If a Parquet file is corrupted, the video is still processed and saved.

### Fixed
- **Parquet Parsing**: Fixed a `TypeError` in `src/joint_processor.py` where nested lists in `info.json` caused failures.
- **Output Consistency**: 
    - Fixed an issue where only the first video of an episode was processed. Now iterates through all video candidates (e.g., multiple camera views).
    - Fixed `AttributeError` in logging statements within `src/pipeline_core.py`.
- **Video Playback**: Resolved issues where processed videos were unplayable by enforcing standard codecs (H.264/AV1) via FFmpeg instead of relying solely on OpenCV's default encoding.
