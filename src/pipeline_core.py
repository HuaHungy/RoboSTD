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
from src.image_mirror import mirror_image_dir
from src.dataset_reconstruction import reconstruct_lerobot_dataset
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
        Order: parquet first, then images (if any).
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            logger.error(f"Input directory {input_path} does not exist.")
            return False

        # Create output directory (exact path, no timestamp subdir)
        final_output_dir = output_path
        final_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {final_output_dir}")

        # Locate meta info for joint mapping
        info_path = input_path / 'meta' / 'info.json'
        if not info_path.exists():
            logger.warning(f"Meta info not found at {info_path}. Joint transformation might fail.")
            field_map = {}
        else:
            field_map = self.joint_processor.get_field_map(str(info_path))

        # 1. Find all parquet files
        parquet_files = list(input_path.glob('data/chunk-*/episode_*.parquet'))
        logger.info(f"Found {len(parquet_files)} episodes.")

        # 2. Determine joint processing mode based on operations
        joint_mode = 'copy'
        op_names = [op['name'] for op in operations]
        if 'mirror' in op_names:
            joint_mode = 'transform'   # mirror joints using rule
        elif 'reverse' in op_names:
            joint_mode = 'reverse'     # time-reverse joints

        # 3. Process parquet files (joints) first
        success_count = 0
        fail_count = 0

        for pq_file in parquet_files:
            episode_id = pq_file.stem                # e.g. 'episode_000000'
            chunk_dir = pq_file.parent.name          # e.g. 'chunk-000'
            rel_pq = pq_file.relative_to(input_path)  # relative path inside dataset
            out_pq = final_output_dir / rel_pq
            out_pq.parent.mkdir(parents=True, exist_ok=True)

            # Search for corresponding videos (optional)
            video_candidates = list(input_path.glob(f"videos/{chunk_dir}/*/{episode_id}.mp4"))

            if not video_candidates:
                # No videos – process joints only
                logger.info(f"No video found for {episode_id} – processing joints only.")
                if self.joint_processor.process_parquet(
                    str(pq_file), str(out_pq),
                    rule_name=self.rule_name,
                    field_map=field_map,
                    mode=joint_mode
                ):
                    logger.info(f"Joints processed: {out_pq}")
                    success_count += 1
                else:
                    logger.error(f"Failed to process joints for {episode_id}")
                    fail_count += 1
                continue

            # Videos exist – process each video together with the joints
            episode_success = True
            for video_file in video_candidates:
                rel_vid = video_file.relative_to(input_path)

                # Mirror operation may swap left/right in video folder names
                if 'mirror' in op_names:
                    parts = list(rel_vid.parts)
                    new_parts = []
                    for part in parts[:-1]:  # directories only
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
                    new_parts.append(parts[-1])  # filename
                    rel_vid = Path(*new_parts)

                out_vid = final_output_dir / rel_vid
                out_vid.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"Processing {episode_id} - {video_file.parent.name}...")
                if not self.process_episode(
                    str(video_file), str(pq_file),
                    str(out_vid), str(out_pq),
                    operations, field_map
                ):
                    episode_success = False
                    logger.error(f"Failed to process video {video_file.name}")

            if episode_success:
                success_count += 1
            else:
                fail_count += 1

        # 4. Mirror images (after all joints are done – if mirror operation exists)
        if 'mirror' in op_names:
            images_dir = input_path / "images"
            if images_dir.exists():
                logger.info("Mirror operation: processing images directory...")
                self._mirror_image_episodes(str(input_path), str(final_output_dir))
            else:
                logger.info("No images directory found, skipping image mirror.")

        # 5. Copy meta files
        try:
            shutil.copytree(input_path / 'meta', final_output_dir / 'meta', dirs_exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not copy meta directory: {e}")

        # 6. Final report
        logger.info(f"Processing complete. Success: {success_count}, Fail: {fail_count}")
        return True

    def reconstruct_dataset(self, input_dir, mirror_dir, output_dir,
                            n_units=None, source_arm=None, language='',
                            planner_name=None, objects=None, workspace=None,
                            action_mode=None, allow_planner_fallback=None,
                            constraints_payload=None, unit_names=None,
                            overwrite=False):
        """Run paper Algorithm 1 and materialize a complete LeRobot dataset."""
        stage2_cfg = self.config.get('stage2', {}) if self.config else {}
        return reconstruct_lerobot_dataset(
            input_dir,
            mirror_dir,
            output_dir,
            n_units=n_units if n_units is not None else stage2_cfg.get('default_n_units', 4),
            source_arm=(source_arm or stage2_cfg.get('source_arm', 'R')).upper(),
            language=language,
            planner_name=planner_name or stage2_cfg.get('planner', 'openai'),
            model=stage2_cfg.get('model', 'gpt-4.1'),
            objects=objects,
            workspace=workspace,
            action_mode=action_mode or stage2_cfg.get('action_mode', 'position'),
            allow_planner_fallback=(
                allow_planner_fallback
                if allow_planner_fallback is not None
                else stage2_cfg.get('allow_planner_fallback', False)
            ),
            constraints_payload=constraints_payload,
            unit_names=unit_names,
            overwrite=overwrite,
        )

    def process_episode(self, video_in, joint_in, video_out, joint_out, operations, field_map):
        """
        Process a single episode (video + joint) through the pipeline.
        """
        temp_files = []
        current_vid = video_in
        current_joint = joint_in

        temp_dir = Path(video_out).parent / "temp"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Detect original codec
            original_codec = self.detect_video_codec(video_in)

            # Pre‑transcode if needed (e.g. AV1)
            if original_codec == 'av1' or self.check_needs_transcoding(video_in):
                preprocessed_vid = temp_dir / "preprocessed_input.mp4"
                logger.info(f"Transcoding {video_in} to {preprocessed_vid} for processing...")
                if self.transcode_video(video_in, str(preprocessed_vid), codec='libx264'):
                    current_vid = str(preprocessed_vid)
                    temp_files.append(preprocessed_vid)
                else:
                    logger.error("Pre‑transcoding failed. Aborting.")
                    return False

            # Apply each operation
            for idx, op in enumerate(operations):
                op_name = op['name']
                params = op.get('params', {})

                step_vid_out = temp_dir / f"step_{idx}.mp4"
                step_joint_out = temp_dir / f"step_{idx}.parquet"

                # Video processing
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
                elif op_name == 'mask_mirror':
                    vid_success = video_mask_mirror.mask_mirror_video(current_vid, str(step_vid_out), **params)
                else:
                    logger.error(f"Unknown operation: {op_name}")
                    return False

                if not vid_success:
                    logger.error(f"Video operation '{op_name}' failed.")
                    return False

                # Joint processing
                joint_mode = 'copy'
                if op_name == 'mirror':
                    joint_mode = 'transform'
                elif op_name == 'reverse':
                    joint_mode = 'reverse'

                if not self.joint_processor.process_parquet(
                    current_joint, str(step_joint_out),
                    rule_name=self.rule_name,
                    field_map=field_map,
                    mode=joint_mode
                ):
                    logger.error(f"Joint operation '{op_name}' failed.")
                    return False

                current_vid = str(step_vid_out)
                current_joint = str(step_joint_out)
                temp_files.extend([step_vid_out, step_joint_out])

            # Final encode with original or improved codec
            target_codec = original_codec
            if target_codec in ('unknown', 'mpeg4'):
                target_codec = 'h264'

            encoder_map = {
                'h264': 'libx264',
                'av1': 'libsvtav1',
                'hevc': 'libx265',
                'vp9': 'libvpx-vp9'
            }
            encoder = encoder_map.get(target_codec, 'libx264')

            final_temp = temp_dir / "final_output.mp4"
            if self.transcode_video(current_vid, str(final_temp), codec=encoder):
                shutil.copy2(str(final_temp), video_out)
            else:
                logger.warning("Final transcode failed, using un‑encoded output.")
                shutil.copy2(current_vid, video_out)

            shutil.copy2(current_joint, joint_out)
            self.verify_checksum(video_out, joint_out)
            return True

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return False
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    # --------------- helper methods ---------------

    def check_needs_transcoding(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return True
        ret, _ = cap.read()
        cap.release()
        return not ret

    def detect_video_codec(self, video_path):
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def transcode_video(self, input_path, output_path, codec="libx264"):
        try:
            cmd = ["ffmpeg", "-y", "-i", input_path, "-c:v", codec, "-pix_fmt", "yuv420p"]
            if codec == "libsvtav1":
                cmd.extend(["-preset", "8", "-crf", "30"])
            elif codec == "libx264":
                cmd.extend(["-preset", "fast", "-crf", "23"])
            cmd.append(output_path)
            res = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg transcoding error ({codec}): {e}\nFFmpeg Stderr:\n{e.stderr}")
            return False
        except Exception as e:
            logger.error(f"FFmpeg transcoding error ({codec}): {e}")
            return False

    def _mirror_image_episodes(self, input_root, output_root, camera_names=None):
        """
        Mirror all frames inside images/ directory (lossless PNG output).
        """
        input_images = Path(input_root) / "images"
        output_images = Path(output_root) / "images"

        if not input_images.exists():
            logger.info("No images directory found, skipping image mirror.")
            return

        if camera_names is None:
            camera_names = [d.name for d in input_images.iterdir() if d.is_dir()]

        for cam in camera_names:
            cam_in = input_images / cam
            if not cam_in.exists():
                logger.warning(f"Camera directory not found: {cam_in}")
                continue

            for ep_dir in sorted(cam_in.glob("episode_*")):
                if not ep_dir.is_dir():
                    continue
                episode_name = ep_dir.name
                out_ep_dir = output_images / cam / episode_name
                out_ep_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"  Mirroring images for camera '{cam}', {episode_name}")
                mirror_image_dir(str(ep_dir), str(out_ep_dir), pattern="frame_*.jpg", lossless=False)

    def verify_checksum(self, video_path, joint_path):
        cap = cv2.VideoCapture(video_path)
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        try:
            df = pd.read_parquet(joint_path)
            joint_rows = len(df)
        except Exception:
            joint_rows = -1

        if video_frames != joint_rows:
            logger.warning(
                f"Frame count mismatch: {video_path} ({video_frames}) vs {joint_path} ({joint_rows})"
            )
            return False
        return True
