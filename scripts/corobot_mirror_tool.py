import os
import shutil
import subprocess
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime

import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mirror_tool.log")
    ]
)
logger = logging.getLogger(__name__)

def get_video_codec(filepath):
    """Detect video codec using ffprobe."""
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "v:0", 
            "-show_entries", "stream=codec_name", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "libx264" # Default fallback

def mirror_video(input_path, output_path):
    """
    Mirror video horizontally using ffmpeg.
    Preserves metadata and attempts to maintain quality.
    """
    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg is unavailable; using the OpenCV MP4 fallback")
        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            logger.error("Could not open %s", input_path)
            return False
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            logger.error("Could not create %s", output_path)
            return False
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(cv2.flip(frame, 1))
        finally:
            capture.release()
            writer.release()
        return True

    try:
        # Detect codec to match input if possible, or default to h264
        codec = get_video_codec(input_path)
        video_encoder = "libx264"
        
        # Map detected codec to ffmpeg encoder name
        if codec == 'hevc':
            video_encoder = 'libx265'
        elif codec == 'av1':
            video_encoder = 'libsvtav1'
        elif codec == 'vp9':
            video_encoder = 'libvpx-vp9'
            
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", "hflip",
            "-c:v", video_encoder,
            "-preset", "fast",
            "-crf", "18", # High quality
            "-c:a", "copy", # Copy audio
            "-map_metadata", "0", # Preserve global metadata
            output_path
        ]
        
        # Adjust params for specific encoders
        if video_encoder == 'libsvtav1':
             cmd.extend(["-crf", "30", "-preset", "8"])
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to mirror {input_path}: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False

def create_backup(source_dir, backup_root):
    """Create a full backup of the source directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(backup_root, f"backup_{timestamp}")
    
    logger.info(f"Creating backup at {backup_dir}...")
    try:
        shutil.copytree(source_dir, backup_dir)
        logger.info("Backup created successfully.")
        return backup_dir
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return None

def restore_backup(backup_dir, source_dir):
    """Restore from backup in case of failure."""
    logger.info(f"Restoring from {backup_dir} to {source_dir}...")
    try:
        if os.path.exists(source_dir):
            shutil.rmtree(source_dir)
        shutil.copytree(backup_dir, source_dir)
        logger.info("Restore successful.")
    except Exception as e:
        logger.error(f"Restore failed: {e}")

def process_directory(root_dir):
    """Main processing logic."""
    root_path = Path(root_dir)
    if not root_path.exists():
        logger.error(f"Directory not found: {root_dir}")
        return False

    # 1. Create Backup
    backup_root = root_path.parent / "backups"
    backup_root.mkdir(exist_ok=True)
    backup_dir = create_backup(root_dir, str(backup_root))
    
    if not backup_dir:
        logger.error("Aborting due to backup failure.")
        return False

    try:
        # Traverse chunks
        chunk_dirs = [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("chunk")]
        
        for chunk in chunk_dirs:
            logger.info(f"Processing chunk: {chunk.name}")
            subdirs = [d for d in chunk.iterdir() if d.is_dir()]
            
            # Categorize directories
            # Priority: Left/Right logic takes precedence over High/Head logic to avoid double processing
            left_dirs = [d for d in subdirs if "left" in d.name.lower()]
            right_dirs = [d for d in subdirs if "right" in d.name.lower()]
            
            # High/Head dirs that are NOT left/right
            high_dirs = [d for d in subdirs if ("high" in d.name.lower() or "head" in d.name.lower()) 
                         and "left" not in d.name.lower() 
                         and "right" not in d.name.lower()]
            
            # Set of processed directories to avoid double processing
            processed_dirs = set()

            # Process High/Head (In-place mirror)
            for d in high_dirs:
                if d in processed_dirs: continue
                logger.info(f"Processing High/Head folder: {d.name}")
                
                videos = list(d.glob("*.mp4"))
                for vid in videos:
                    temp_out = vid.with_suffix(".temp.mp4")
                    if mirror_video(str(vid), str(temp_out)):
                        # Replace original
                        os.replace(temp_out, vid)
                    else:
                        raise Exception(f"Failed to mirror {vid}")
                processed_dirs.add(d)

            # Process Left/Right Pairs (Swap)
            for d_left in left_dirs:
                if d_left in processed_dirs: continue
                
                # Find matching right
                right_name = d_left.name.replace("left", "right")
                if "Left" in d_left.name:
                    right_name = d_left.name.replace("Left", "Right")
                elif "LEFT" in d_left.name:
                    right_name = d_left.name.replace("LEFT", "RIGHT")
                    
                d_right = chunk / right_name
                
                if d_right.exists() and d_right.is_dir():
                    logger.info(f"Processing Pair: {d_left.name} <-> {d_right.name}")
                    
                    # Create temp dirs for swapping
                    temp_left_dir = chunk / f"{d_left.name}_temp"
                    temp_right_dir = chunk / f"{d_right.name}_temp"
                    if temp_left_dir.exists(): shutil.rmtree(temp_left_dir)
                    if temp_right_dir.exists(): shutil.rmtree(temp_right_dir)
                    temp_left_dir.mkdir(exist_ok=True)
                    temp_right_dir.mkdir(exist_ok=True)
                    
                    try:
                        # Mirror Left -> Temp Right
                        # Content from LEFT folder is mirrored and moved to RIGHT folder
                        for vid in d_left.glob("*.mp4"):
                            out_path = temp_right_dir / vid.name
                            if not mirror_video(str(vid), str(out_path)):
                                raise Exception(f"Failed to mirror left video {vid}")
                                
                        # Mirror Right -> Temp Left
                        # Content from RIGHT folder is mirrored and moved to LEFT folder
                        for vid in d_right.glob("*.mp4"):
                            out_path = temp_left_dir / vid.name
                            if not mirror_video(str(vid), str(out_path)):
                                raise Exception(f"Failed to mirror right video {vid}")
                        
                        # Apply Swap: 
                        # Overwrite Left with Temp Left (which contains mirrored Right content)
                        # Overwrite Right with Temp Right (which contains mirrored Left content)
                        
                        # Clear original folders (Instead of deleting folders, we just delete files inside)
                        # Wait, user said "only one folder remains". 
                        # If os.remove fails or shutil.move fails, we might end up with empty folders or deleted files.
                        # But if we delete files, the folder structure should remain.
                        
                        # Use shutil.rmtree to clear content then recreate? No, just delete files.
                        for vid in d_left.glob("*.mp4"): os.remove(vid)
                        for vid in d_right.glob("*.mp4"): os.remove(vid)
                        
                        # Move temp files to original folders
                        for vid in temp_left_dir.glob("*.mp4"):
                            shutil.move(str(vid), d_left / vid.name)
                        for vid in temp_right_dir.glob("*.mp4"):
                            shutil.move(str(vid), d_right / vid.name)
                    finally:
                        # Cleanup temp dirs
                        if temp_left_dir.exists(): shutil.rmtree(temp_left_dir)
                        if temp_right_dir.exists(): shutil.rmtree(temp_right_dir)
                    
                    processed_dirs.add(d_left)
                    processed_dirs.add(d_right)
                else:
                    logger.warning(f"No matching right folder found for {d_left.name}. Skipping exchange.")

        logger.info("All processing completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Critical error during processing: {e}")
        logger.info("Initiating rollback...")
        restore_backup(backup_dir, root_dir)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoRobot Video Mirror Tool")
    parser.add_argument("--root", required=True, help="Root directory of videos")
    args = parser.parse_args()
    
    process_directory(args.root)
