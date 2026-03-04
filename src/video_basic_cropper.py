import cv2
import numpy as np

def crop_video_half(video_path, output_path, axis='vertical', keep_side='first'):
    """
    Crops video symmetrically along the axis (half), filling the cropped part with a white background.
    
    Args:
        video_path (str): Path to input video.
        output_path (str): Path to save output video.
        axis (str): 'vertical' (split left/right) or 'horizontal' (split top/bottom).
        keep_side (str): 'first' (left/top) or 'second' (right/bottom).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a white background frame of the same size
        # numpy array: (height, width, channels)
        # 255 is white
        white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Determine the region to keep and copy it to the white background
        if axis == 'vertical':
            mid_x = width // 2
            if keep_side == 'first': # Keep left
                white_bg[:, :mid_x] = frame[:, :mid_x]
            else: # Keep right
                white_bg[:, mid_x:] = frame[:, mid_x:]
        elif axis == 'horizontal':
            mid_y = height // 2
            if keep_side == 'first': # Keep top
                white_bg[:mid_y, :] = frame[:mid_y, :]
            else: # Keep bottom
                white_bg[mid_y:, :] = frame[mid_y:, :]
        
        out.write(white_bg)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}...", end='\r')

    cap.release()
    out.release()
    print(f"\nCropping complete: {output_path}")
    return True
