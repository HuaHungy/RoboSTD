import cv2

def convert_video(input_path, output_path, target_width=None, target_height=None, target_fps=None):
    """
    Converts MP4 video to specified resolution and frame rate.
    
    Args:
        input_path (str): Path to input video.
        output_path (str): Path to save output video.
        target_width (int, optional): Target width. Defaults to None (keep original).
        target_height (int, optional): Target height. Defaults to None (keep original).
        target_fps (int, optional): Target frame rate. Defaults to None (keep original).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False

    # Original properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)

    # Determine output properties
    width = target_width if target_width else orig_width
    height = target_height if target_height else orig_height
    fps = target_fps if target_fps else orig_fps

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use mp4v for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize if needed
        if width != orig_width or height != orig_height:
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
        frame_count += 1
        
        # Simple progress indicator
        if frame_count % 30 == 0:
            print(f"Converting frame {frame_count}...", end='\r')

    cap.release()
    out.release()
    print(f"\nConversion complete: {output_path}")
    return True
