# RoboSTD Unified Project

## Introduction
RoboSTD Unified is a comprehensive toolkit for processing robotic datasets, specifically designed for the CoRobot format. It integrates video processing (cropping, mirroring, stitching, etc.) with synchronous joint data transformation.

## Directory Structure
- `src/`: Core logic modules (`video_*`, `joint_*`).
- `scripts/`: Executable scripts (`main.py`, `corobot_mirror_tool.py`).
- `configs/`: Configuration files (`default.yaml`).
- `tools/`: Verification and testing tools.
- `data/`: Dataset storage (input/output).

## Installation
1. Clone the repository.
2. Install system dependencies (Required for AV1 video support):
   ```bash
   sudo apt-get install ffmpeg
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start
### CoRobot Mirror Tool
A specialized tool for mirroring CoRobot dataset videos, including automatic swapping of left/right camera views.

```bash
python scripts/corobot_mirror_tool.py --root /path/to/videos
```
Features:
- **Mirroring**: Horizontally flips videos in `high` or `head` folders.
- **Mirror & Swap**: Mirrors videos in `left` and `right` folders, then swaps their contents.
- **Backup**: Automatically backs up the original directory before processing.
- **Rollback**: Restores from backup in case of critical errors.

### Interactive Mode
Run the main script without arguments to start the interactive wizard:
```bash
python scripts/main.py
```

### Command Line Mode
Run the pipeline with specific arguments:
```bash
python scripts/main.py \
  --input data/raw \
  --output data/output \
  --ops mirror,reverse \
  --rule aloha
```

## Configuration
The `configs/default.yaml` file defines:
- **Joint Mirror Rules**: Mappings for flipping left/right arms (e.g., `aloha`, `realman`).
- **Video Settings**: Default resolution/FPS.

## API Documentation
### Video Modules
- `video_mirror.mirror_video(input, output)`: Flips video horizontally.
- `video_reverse.reverse_video(input, output)`: Reverses video playback.
- `video_converter.convert_video(input, output, width, height, fps)`: Resizes/Resamples video.
- ... (See `src/` for all modules)

### Joint Processor
- `JointProcessor.process_parquet(src, dst, rule, field_map, mode)`: Transforms joint data.
  - `mode='transform'`: Applies mirror rules.
  - `mode='reverse'`: Reverses rows.
  - `mode='copy'`: Identity.

## Contribution
1. Fork the repository.
2. Create a feature branch.
3. Submit a Pull Request.
4. Ensure `tools/verify_project.sh` passes.

## License
MIT License
