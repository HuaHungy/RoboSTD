import argparse
import logging
import os

# Suppress Rerun warnings
os.environ["RERUN_LOG"] = "error"
logging.getLogger("rerun").setLevel(logging.ERROR)

import sys
import time
import traceback
from pathlib import Path
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

# Try importing av for robust AV1 decoding
try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

# 将项目根目录加入路径，确保直接运行时导入正常
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  # 从 src/replay 回到项目根目录
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.replay.SimReplayer import LerobotSimReplayer

def get_error_frame(width=640, height=480, text="Error"):
    """Generate a black frame with error text"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 255, 255) # White
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
    return img

class VideoReader:
    """Wrapper to handle video reading with fallback (AV -> CV2 -> Error Frame)"""
    def __init__(self, video_path):
        self.video_path = str(video_path)
        self.use_av = HAS_AV
        self.cap = None
        self.container = None
        self.stream = None
        self.frame_iter = None
        self.width = 640
        self.height = 480
        self.is_image = False
        self._init_video()
    
    def _init_video(self):
        if self.use_av:
            try:
                self.container = av.open(self.video_path)
                self.stream = self.container.streams.video[0]
                self.stream.thread_type = "AUTO" # Enable multi-threading
                self.frame_iter = self.container.decode(self.stream)
                # Get dimensions
                self.width = self.stream.width
                self.height = self.stream.height
            except Exception as e:
                print(f"PyAV failed to open {self.video_path}: {e}\n{traceback.format_exc()}. Falling back to OpenCV.")
                self.use_av = False
        
        if not self.use_av:
            self.cap = cv2.VideoCapture(self.video_path)
            if self.cap.isOpened():
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def read(self):
        try:
            if self.use_av:
                try:
                    frame = next(self.frame_iter)
                    # Convert to numpy array (RGB)
                    img = frame.to_ndarray(format="rgb24")
                    return True, img
                except StopIteration:
                    return False, None
                except Exception as e:
                    print(f"PyAV decoding error: {e}\n{traceback.format_exc()}")
                    return False, None
            else:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        # OpenCV returns BGR, convert to RGB
                        return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return False, None
                return False, None
        except Exception as e:
            print(f"Error reading frame: {e}\n{traceback.format_exc()}")
            return False, None

    def release(self):
        if self.use_av and self.container:
            self.container.close()
        if self.cap:
            self.cap.release()


class ImageReader:
    """Wrapper to handle image sequence loading from directory"""
    def __init__(self, image_dir, image_key):
        self.image_dir = Path(image_dir)
        self.image_key = image_key
        self.frame_idx = 0
        self.width = 640
        self.height = 480
        self.total_frames = 0
        self._episode_idx = -1          # 初始化为 -1，等待 set_episode 赋值
        # 不再此处调用 _init_images，由 set_episode 触发

    def _init_images(self):
        """Initialize image sequence - find all frames for this episode"""
        if self._episode_idx < 0:
            print("Warning: episode_idx not set, cannot init images.")
            return

        episode_pattern = f"episode_{self._episode_idx:06d}"
        self.episode_dir = self.image_dir / episode_pattern

        if not self.episode_dir.exists():
            print(f"Warning: Image directory not found: {self.episode_dir}")
            return

        # 按文件名排序，确保顺序正确
        self.frame_files = sorted(
            list(self.episode_dir.glob("frame_*.jpg")) + 
            list(self.episode_dir.glob("frame_*.png"))
        )
        self.total_frames = len(self.frame_files)

        if self.total_frames > 0:
            first_img = cv2.imread(str(self.frame_files[0]))
            if first_img is not None:
                self.height, self.width = first_img.shape[:2]

    def set_episode(self, episode_idx):
        """Set the episode index and reinitialize"""
        self._episode_idx = episode_idx
        self.frame_idx = 0
        self._init_images()

    def read(self):
        """Read next frame from image sequence"""
        if self.frame_idx >= self.total_frames:
            return False, None

        try:
            frame_path = self.frame_files[self.frame_idx]
            img = cv2.imread(str(frame_path))

            if img is None or img.size == 0:
                print(f"Error: Failed to read image: {frame_path}")
                return False, None

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.frame_idx += 1
            return True, img_rgb

        except Exception as e:
            print(f"Error reading image frame {self.frame_idx}: {e}")
            return False, None

    def release(self):
        pass

def run_replay(repo_path, config_name, data_source="data", data_type="all", episode_idx=0, auto_close=True, version="version"):
    # 解析配置文件路径
    config_path = project_root / "configs" / "sim_replay" / f"{config_name}.yml"
    
    if not config_path.exists():
        # 兜底：用户可能传入了完整文件名
        if not config_name.endswith(".yml"):
            config_path_alt = project_root / "configs" / "sim_replay" / f"{config_name}"
            if config_path_alt.exists():
                config_path = config_path_alt
        
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Using config: {config_path} with version: {version}")
    
    # 使用 repo_path 的最后一段作为 Rerun 标题
    repo_name = Path(repo_path).name
    rr.init(repo_name)
    rr.spawn()
    
    # 初始化 SimReplayer
    replayers = {}
    if data_type == "all":
        print("Initializing State Replayer...")
        replayers["state"] = LerobotSimReplayer(
            config_path=str(config_path),
            version=version,
            repo_path=repo_path,
            episode_idx=episode_idx,
            data_source=data_source,
            data_type="state",
            headless=True,
            render_width=640,
            render_height=480
        )
        print("Initializing Action Replayer...")
        replayers["action"] = LerobotSimReplayer(
            config_path=str(config_path),
            version=version,
            repo_path=repo_path,
            episode_idx=episode_idx,
            data_source=data_source,
            data_type="action",
            headless=True,
            render_width=640,
            render_height=480
        )
    else:
        print(f"Initializing {data_type} Replayer...")
        replayers[data_type] = LerobotSimReplayer(
            config_path=str(config_path),
            version=version,
            repo_path=repo_path,
            episode_idx=episode_idx,
            data_source=data_source,
            data_type=data_type,
            headless=True,
            render_width=640,
            render_height=480
        )
    
    # 打开视频或图片
    # 路径格式：videos/chunk-num/camera_name/episode_num.mp4
    # 或：images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpg
    chunk_idx = episode_idx // 1000
    chunk_dir_name = f"chunk-{chunk_idx:03d}"
    video_chunk_path = Path(repo_path) / "videos" / chunk_dir_name
    image_chunk_path = Path(repo_path) / "images"
    
    video_readers = {}
    image_readers = {}
    
    # 首先尝试加载视频
    if video_chunk_path.exists():
        print(f"Searching for videos in: {video_chunk_path}")
        for cam_dir in video_chunk_path.iterdir():
            if cam_dir.is_dir():
                # 检查视频文件
                video_path = cam_dir / f"episode_{episode_idx:06d}.mp4"
                if video_path.exists():
                    print(f"Found video: {video_path}")
                    # Use VideoReader wrapper
                    reader = VideoReader(video_path)
                    video_readers[cam_dir.name] = reader
                else:
                    print(f"Video not found in {cam_dir}: {video_path}")
    else:
        print(f"Warning: Video chunk directory not found at {video_chunk_path}")
    
    # 如果没有视频，尝试加载图片
    if not video_readers and image_chunk_path.exists():
        print(f"Searching for images in: {image_chunk_path}")
        # 直接扫描 images 目录下的所有子目录，每个子目录即一个相机源
        for subdir in sorted(image_chunk_path.iterdir()):
            if subdir.is_dir():
                cam_name = subdir.name  # 完整名称，如 observation.images.image_left
                img_dir = subdir
                print(f"Found image directory: {img_dir}")
                reader = ImageReader(img_dir, cam_name)
                reader.set_episode(episode_idx)
                if reader.total_frames > 0:
                    image_readers[cam_name] = reader
                    print(f"Found {reader.total_frames} frames for {cam_name}")
                else:
                    print(f"Warning: No frames found in {img_dir}")
    # 合并 video_readers 和 image_readers
    all_media_readers = {**video_readers, **image_readers}

    # Define layout keywords mapping
    def get_layout_pos(name):
        name_lower = name.lower()
        
        # Column: 0=Left, 1=Center, 2=Right
        col = 1
        if "left" in name_lower:
            col = 0
        elif "right" in name_lower:
            col = 2
            
        # Row: 0=Top, 1=Middle, 2=Bottom
        row = 1
        
        top_kws = ["head", "top", "upper", "global", "env"]
        bottom_kws = ["leg", "lower", "bottom", "wrist", "foot"]
        
        for kw in top_kws:
            if kw in name_lower:
                row = 0
                break
        
        if row == 1:
            for kw in bottom_kws:
                if kw in name_lower:
                    row = 2
                    break
        
        return col, row

    if not all_media_readers:
        video_container = rrb.Spatial2DView(origin="/01_videos", name="Videos")
    else:
        # Grid: [col][row] -> list of views
        grid = [[[] for _ in range(3)] for _ in range(3)]
        
        for cam_name in sorted(all_media_readers.keys()):
            col, row = get_layout_pos(cam_name)
            # view = rrb.Spatial2DView(origin=f"/01_videos/{cam_name}", name=cam_name.split(".")[-1]) # Use simpler name if possible
            view = rrb.Spatial2DView(origin=f"/01_videos/{cam_name}/image", name=cam_name)
            grid[col][row].append(view)
            
        # Build columns
        cols = []
        # Explicitly iterate 0, 1, 2 to maintain Left-Center-Right order
        for c in range(3):
            col_views = []
            for r in range(3):
                col_views.extend(grid[c][r])
            
            if col_views:
                if len(col_views) > 1:
                    cols.append(rrb.Vertical(*col_views))
                else:
                    cols.append(col_views[0])
            
        if not cols:
            video_container = rrb.Spatial2DView(origin="/01_videos", name="Videos")
        elif len(cols) == 1:
            video_container = cols[0]
        else:
            video_container = rrb.Horizontal(*cols)

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                video_container,
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="/02_state/image", name="State"),
                    rrb.Spatial2DView(origin="/03_action/image", name="Action"),
                ),
            ),
            auto_layout=False,
        )
    )
    
    print(f"Loading data into Rerun for episode {episode_idx}...")
    
    frame_idx = 0
    # dt = 0.01  # 假设 50Hz，可按需调整
    try:
        while True:
            # 推进所有 replayer
            steps_ok = True
            for name, replayer in replayers.items():
                if not replayer.step():
                    steps_ok = False
                    break
            
            if not steps_ok:
                break
            
            rr.set_time_sequence("frame_index", frame_idx)
            
            # 1. 记录视频或图片（加前缀保证顺序）
            for cam_name, reader in all_media_readers.items():
                ret, frame_rgb = reader.read()
                
                if not ret or frame_rgb is None:
                    # Generate error frame
                    frame_rgb = get_error_frame(width=reader.width, height=reader.height, text="Error")
                    
                # 直接写入相机名（父目录名），使用前缀保证顺序
                rr.log(f"01_videos/{cam_name}/image", rr.Image(frame_rgb))
            
            # 2. 记录 MuJoCo 图像（前缀区分 state/action）
            for name, replayer in replayers.items():
                mj_img = replayer.get_img()
                if mj_img is None:
                    continue
                
                prefix = ""
                if name == "state":
                    prefix = "02_state"
                elif name == "action":
                    prefix = "03_action"
                else:
                    prefix = f"04_{name}"
                    
                rr.log(f"{prefix}/image", rr.Image(mj_img))
                
            
            frame_idx += 1
            
            # 降速以便实时查看，避免瞬间播放完
            # time.sleep(dt)
            
    except KeyboardInterrupt:
        print("Replay interrupted by user.")
    except Exception as e:
        error_msg = f"Error during replay loop: {e}\n{traceback.format_exc()}"
        print(error_msg)
        raise RuntimeError(error_msg) from e
    finally:
        # Release resources
        for reader in video_readers.values():
            reader.release()
        
        # Cleanup replayers if needed (though they mainly just hold data)
        pass

        if auto_close:
            # Give some time for Rerun to send data
            time.sleep(1.0) 
            try:
                rr.disconnect()
            except Exception:
                pass
            # Note: Rerun spawns a separate process/tab, we can't force close it easily from here
            # but we can stop the script.
            print("Replay finished.")
