import os
import shutil
import glob
import time
import logging
import cv2
import pandas as pd
import subprocess
from datetime import datetime
from pathlib import Path

# Import our modules
from src.joint_processor import JointProcessor
from src import video_converter, video_mirror, video_reverse, video_basic_cropper, video_advanced_cropper, video_stitcher, video_mask_mirror, video_inspector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, config_path, rule_name):
        self.joint_processor = JointProcessor(config_path)
        self.rule_name = rule_name
        self.config = self.joint_processor.config

    def process_dataset(self, input_dir, output_dir, operations):
        """
        Process the entire dataset with the given operations.
        
        Args:
            input_dir (str): Root of the dataset.
            output_dir (str): Root of the output.
            operations (list): List of operation dictionaries, e.g. [{'name': 'mirror'}, {'name': 'reverse'}]
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            logger.error(f"Input directory {input_path} does not exist.")
            return False

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_dir = output_path / timestamp
        final_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created at {final_output_dir}")

        # Locate meta info for joint mapping
        info_path = input_path / 'meta' / 'info.json'
        if not info_path.exists():
            logger.warning(f"Meta info not found at {info_path}. Joint transformation might fail.")
            field_map = {}
        else:
            field_map = self.joint_processor.get_field_map(str(info_path))

        # Find episodes (assuming LeRobot structure: data/chunk-*/episode_*.parquet)
        # and videos (videos/chunk-*/observation.../episode_*.mp4)
        # We need to match them.
        
        # 1. Find all parquet files
        parquet_files = list(input_path.glob('data/chunk-*/episode_*.parquet'))
        logger.info(f"Found {len(parquet_files)} episodes.")

        # 2. Find corresponding videos
        # Usually video structure matches data structure but in 'videos' folder.
        # But video filenames might have extensions or be in subfolders like 'observation.images.cam_high_rgb'
        # We need to search for video files with the same episode_id.
        
        success_count = 0
        fail_count = 0

        for pq_file in parquet_files:
            episode_id = pq_file.stem # 'episode_000000'
            chunk_dir = pq_file.parent.name # 'chunk-000'
            
            # Search for video
            # Pattern: input_dir/videos/chunk-*/observation.*/episode_*.mp4
            # We assume the chunk matches.
            video_candidates = list(input_path.glob(f"videos/{chunk_dir}/*/{episode_id}.mp4"))
            
            if not video_candidates:
                logger.warning(f"No video found for {episode_id} in {chunk_dir}. Skipping.")
                fail_count += 1
                continue
            
            # Process ALL found videos for this episode
            episode_success = True
            
            # We process parquet only once (or overwrite it identically)
            # To avoid redundant processing, we could process it once and then just copy,
            # but for simplicity and to ensure sync logic is maintained per-video pipeline,
            # we will allow overwriting.
            
            for video_file in video_candidates:
                # Define output paths
                # Maintain relative structure
                rel_pq = pq_file.relative_to(input_path)
                rel_vid = video_file.relative_to(input_path)
                
                # Check for mirror operation and swap left/right folder names if needed
                op_names = [op['name'] for op in operations]
                if 'mirror' in op_names:
                    parts = list(rel_vid.parts)
                    new_parts = []
                    for part in parts[:-1]: # Iterate over directories only, exclude filename
                        lower_part = part.lower()
                        new_part = part
                        if 'left' in lower_part:
                            if 'left' in part: new_part = part.replace('left', 'right')
                            elif 'Left' in part: new_part = part.replace('Left', 'Right')
                            elif 'LEFT' in part: new_part = part.replace('LEFT', 'RIGHT')
                        elif 'right' in lower_part:
                            if 'right' in part: new_part = part.replace('right', 'left')
                            elif 'Right' in part: new_part = part.replace('Right', 'Left')
                            elif 'RIGHT' in part: new_part = part.replace('RIGHT', 'LEFT')
                        new_parts.append(new_part)
                    
                    new_parts.append(parts[-1]) # Add filename back
                    rel_vid = Path(*new_parts)

                out_pq = final_output_dir / rel_pq
                out_vid = final_output_dir / rel_vid
                
                out_pq.parent.mkdir(parents=True, exist_ok=True)
                out_vid.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Processing {episode_id} - {video_file.parent.name}...")
                
                if not self.process_episode(str(video_file), str(pq_file), str(out_vid), str(out_pq), operations, field_map):
                    episode_success = False
                    logger.error(f"Failed to process video {video_file.name}")
            
            if episode_success:
                success_count += 1
            else:
                fail_count += 1

        # Copy meta files
        try:
            shutil.copytree(input_path / 'meta', final_output_dir / 'meta')
        except Exception as e:
            logger.warning(f"Could not copy meta directory: {e}")

        logger.info(f"Processing complete. Success: {success_count}, Fail: {fail_count}")
        return True

    def process_episode(self, video_in, joint_in, video_out, joint_out, operations, field_map):
        """
        Process a single episode (video + joint) through the pipeline.
        """
        temp_files = []
        current_vid = video_in
        current_joint = joint_in
        
        # Create temp dir for intermediate steps
        temp_dir = Path(video_out).parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # --- Detect Original Codec ---
            original_codec = self.detect_video_codec(video_in)
            needs_restore_codec = False

            # --- Pre-check Video Encoding (Handle AV1) ---
            preprocessed_vid = temp_dir / "preprocessed_input.mp4"
            
            # If codec is av1, we assume we need to transcode for OpenCV processing
            if original_codec == 'av1' or self.check_needs_transcoding(video_in):
                logger.info(f"Video {video_in} (codec: {original_codec}) requires processing transcoding. Transcoding to {preprocessed_vid}...")
                if self.transcode_video(video_in, str(preprocessed_vid), codec='libx264'):
                    current_vid = str(preprocessed_vid)
                    temp_files.append(preprocessed_vid)
                    # If original was av1, mark for restoration
                    if original_codec == 'av1':
                        needs_restore_codec = True
                else:
                    logger.error("Transcoding failed. Aborting episode.")
                    return False
            
            for idx, op in enumerate(operations):
                op_name = op['name']
                params = op.get('params', {})
                
                step_vid_out = temp_dir / f"step_{idx}.mp4"
                step_joint_out = temp_dir / f"step_{idx}.parquet"
                
                # --- Video Processing ---
                vid_success = False
                if op_name == 'mirror':
                    vid_success = video_mirror.mirror_video(current_vid, str(step_vid_out))
                elif op_name == 'reverse':
                    vid_success = video_reverse.reverse_video(current_vid, str(step_vid_out))
                elif op_name == 'convert':
                    vid_success = video_converter.convert_video(current_vid, str(step_vid_out), **params)
                elif op_name == 'crop_half':
                    vid_success = video_basic_cropper.crop_video_half(current_vid, str(step_vid_out), **params)
                elif op_name == 'crop_advanced':
                    vid_success = video_advanced_cropper.crop_video_advanced(current_vid, str(step_vid_out), **params)
                elif op_name == 'stitch':
                    # Special case: stitch requires 2 videos. We might need another arg or assume stitching with itself?
                    # For now, let's assume we skip stitch in single-episode pipeline unless provided.
                    logger.warning("Stitch operation not fully supported in single-episode pipeline yet.")
                    vid_success = False
                elif op_name == 'mask_mirror':
                    vid_success = video_mask_mirror.mask_mirror_video(current_vid, str(step_vid_out), **params)
                else:
                    logger.error(f"Unknown operation: {op_name}")
                
                if not vid_success:
                    logger.error(f"Video operation {op_name} failed.")
                    return False
                
                # --- Joint Processing ---
                joint_mode = 'copy'
                if op_name == 'mirror':
                    joint_mode = 'transform' # Uses rule
                elif op_name == 'reverse':
                    joint_mode = 'reverse'
                
                # For crop/convert/stitch(primary), we use 'copy' (identity)
                
                joint_success = self.joint_processor.process_parquet(
                    current_joint, 
                    str(step_joint_out), 
                    rule_name=self.rule_name, 
                    field_map=field_map, 
                    mode=joint_mode
                )
                
                if not joint_success:
                    logger.error(f"Joint operation {op_name} failed.")
                    # Keep proceeding with video if joint fails? No, integrity is key.
                    # But if we fail here, we must clean up?
                    return False
                
                # Update current paths for next step
                current_vid = str(step_vid_out)
                current_joint = str(step_joint_out)
                temp_files.append(step_vid_out)
                temp_files.append(step_joint_out)

            # --- Final Transcode / Restore ---
            # Always try to match original codec or default to H.264
            # We skip 'needs_restore_codec' logic and apply global codec policy
            
            target_codec = original_codec
            if target_codec == 'unknown' or target_codec == 'mpeg4': 
                # If unknown or mpeg4 (which is likely what OpenCV produced if not configured well),
                # upgrade to h264 for compatibility.
                target_codec = 'h264'
            
            # Map codec name to ffmpeg encoder
            encoder_map = {
                'h264': 'libx264',
                'av1': 'libsvtav1', # or libaom-av1
                'hevc': 'libx265',
                'vp9': 'libvpx-vp9'
            }
            encoder = encoder_map.get(target_codec, 'libx264')
            
            logger.info(f"Finalizing video {Path(video_out).name} with codec {target_codec} ({encoder})...")
            
            final_temp = temp_dir / "final_output.mp4"
            
            # Force transcode to final location in temp first
            if self.transcode_video(current_vid, str(final_temp), codec=encoder):
                # Now copy to final destination
                shutil.copy2(str(final_temp), video_out)
            else:
                logger.warning(f"Final transcoding to {target_codec} failed. Moving intermediate result.")
                shutil.copy2(current_vid, video_out)
                
            # Copy final joint parquet to destination
            shutil.copy2(current_joint, joint_out)
            
            # --- Checksum ---
            self.verify_checksum(video_out, joint_out)
            
            return True

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return False
        finally:
            # Cleanup temp
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def check_needs_transcoding(self, video_path):
        """
        Check if video needs transcoding (fallback method using OpenCV).
        Use detect_video_codec first to avoid error logs.
        """
        # Suppress OpenCV error output for this check?
        # It's hard to suppress C++ level logs from Python reliably without redirections.
        # But if detect_video_codec works, this shouldn't be called for AV1.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return True
        ret, _ = cap.read()
        cap.release()
        return not ret

    def detect_video_codec(self, video_path):
        """
        Detect video codec using ffprobe to avoid OpenCV error logs.
        """
        try:
            cmd = [
                "ffprobe", 
                "-v", "error", 
                "-select_streams", "v:0", 
                "-show_entries", "stream=codec_name", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            codec = result.stdout.strip()
            return codec
        except Exception:
            return "unknown"

    def transcode_video(self, input_path, output_path, codec="libx264"):
        """
        Transcode video using system ffmpeg.
        """
        try:
            cmd = [
                "ffmpeg", "-y", "-i", input_path, 
                "-c:v", codec, 
                "-c:a", "aac", 
                output_path
            ]
            # Add specific params for libsvtav1 if needed for speed
            if codec == "libsvtav1":
                cmd.extend(["-preset", "8", "-crf", "30"]) # Preset 8 is faster
            elif codec == "libx264":
                cmd.extend(["-preset", "fast", "-crf", "23"])

            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            logger.error(f"FFmpeg transcoding error ({codec}): {e}")
            return False

    def verify_checksum(self, video_path, joint_path):
        """
        Verify frame count matches joint rows.
        """
        cap = cv2.VideoCapture(video_path)
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        try:
            df = pd.read_parquet(joint_path)
            joint_rows = len(df)
        except:
            joint_rows = -1
            
        if video_frames != joint_rows:
            logger.warning(f"CHECKSUM MISMATCH: {video_path} ({video_frames} frames) vs {joint_path} ({joint_rows} rows).")
            # TODO: Implement padding logic if needed
            return False
        return True
