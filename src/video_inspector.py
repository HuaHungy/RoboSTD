import cv2
import os

def inspect_video(video_path):
    """
    Detects video encoding, resolution, dimensions, frame rate, etc.
    
    Args:
        video_path (str): Path to the input video.
        
    Returns:
        dict: A dictionary containing video metadata.
    """
    if not os.path.exists(video_path):
        return {"error": "File not found"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # Convert FOURCC to string
    try:
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    except:
        codec = "unknown"

    duration = frame_count / fps if fps > 0 else 0

    info = {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "codec": codec,
        "duration_sec": duration
    }

    cap.release()
    return info
