# Development Log

## 2026-03-04
### New Feature: CoRobot Mirror Tool
- **Goal**: Implement a specialized video mirroring tool for CoRobot datasets that handles specific folder naming conventions and left/right view swapping.
- **Implementation**:
  - Created `scripts/corobot_mirror_tool.py`.
  - Implemented logic to traverse directory chunks.
  - **Mirror Logic**:
    - Folders with "high" or "head": In-place horizontal flip.
    - Folders with "left" or "right": Horizontal flip followed by swapping content between paired folders (e.g., `cam_left` content moves to `cam_right` after flipping).
  - **Safety**: Added automatic full backup creation before processing and rollback mechanism on failure.
  - **Quality**: Uses `ffmpeg` with CRF 18 (visually lossless) and codec detection to preserve original format (e.g. H.265, AV1).
- **Verification**:
  - Created `tools/tests/test_mirror_tool.py` to simulate the directory structure and verify the mirror/swap logic using dummy videos.

## 2026-03-02 (Update 2)
### Bug Fix: AV1 Codec Preservation & Error Log Suppression
- **Issue**:
  - The pipeline produced excessive error logs (`[av1 @ ...] Failed to get pixel format`) because `check_needs_transcoding` used OpenCV to probe files, triggering stderr output from the C++ backend.
  - The output video was encoded in H.264/MPEG-4, but the user required the output format to align with the original video (AV1).
- **Solution**:
  - **Codec Detection**: Replaced OpenCV probe with `ffprobe` (via `subprocess`) to detect the codec name silently.
  - **Log Suppression**: Only invoking transcoding if `ffprobe` detects 'av1' or if OpenCV truly fails (as a fallback).
  - **Codec Restoration**: Implemented a post-processing step. If the original video was AV1, the pipeline now transcodes the final processed video *back* to AV1 (using `libsvtav1` encoder) before saving it to the output directory.
- **Verification**:
  - Updated `tools/tests/test_av1_fix.py` to verify:
    1. The output file exists.
    2. The output codec is explicitly 'av1'.
    3. The output file is valid (can be transcoded back to H.264).
    4. Frame counts match the original data.
  - Test passed successfully with `Success! Output video valid. Frame count: 235`.

## 2026-03-02 (Update)
### Bug Fix: AV1 Video Support
- **Issue**: Users reported empty output videos and "CHECKSUM MISMATCH" errors. Logs indicated OpenCV failed to decode AV1 encoded videos (`[av1 @ ...] Failed to get pixel format`).
- **Diagnosis**: The installed OpenCV bindings lack proper AV1 hardware/software decoding support, even though the system `ffmpeg` supports it.
- **Solution**:
  - Enhanced `src/pipeline_core.py` with a `check_needs_transcoding` method.
  - Implemented automatic transcoding using system `ffmpeg` (`subprocess`) to convert unreadable/AV1 videos to a temporary H.264 intermediate file before processing.
  - This ensures compatibility with standard OpenCV builds without requiring complex environment reconfiguration.
- **Verification**:
  - Created `tools/tests/test_av1_fix.py` using a sample AV1 video from the dataset.
  - Verified that the pipeline now successfully transcodes, processes (mirrors), and passes the checksum verification (Frame Count == Parquet Rows).

## 2026-03-02
### Initial Refactoring & Integration
- **Goal**: Integrate `joint_process` and `video_editor` into `RoboSTD_Unified`.
- **Structure**: Created standard directory layout (`src`, `scripts`, `configs`, `tools`, `data`).
- **Migration**:
  - Ported `video_editor/video_tools/*.py` to `src/video_*.py`.
  - Refactored `joint_process/process_lerobot.py` into `src/joint_processor.py` class.
- **New Features**:
  - Implemented `src/pipeline_core.py` to handle dataset iteration and synchronous processing.
  - Implemented `scripts/main.py` with both CLI and Interactive modes.
  - Added checksum verification (frame count vs joint rows).
  - Added `tools/verify_project.sh` for one-click verification.
- **Configuration**:
  - Created `configs/default.yaml` merging mirror rules and general settings.
- **Testing**:
  - Added `tools/tests/test_pipeline.py` covering mirror/reverse logic with dummy data.

### Technical Decisions
- **Modularity**: Decoupled video and joint logic. The pipeline acts as the coordinator.
- **Joint Mapping**: Used the existing configuration schema from `joint_process` but extended it to support different "modes" (transform, reverse, copy).
- **Video Tools**: Kept the OpenCV-based implementations but cleaned up imports and interfaces.
- **Interactive Mode**: Replicated the CLI menu experience in `scripts/main.py` while keeping the core logic separate.

### Future Work
- Support for "Stitch" operation with multi-camera joint handling.
- Advanced "Mask Mirror" automation (currently requires GUI).
- Optimization for large video processing (currently `reverse` loads all frames to memory).
