import cv2
import numpy as np

def stitch_videos(video1_path, video2_path, output_path, direction='horizontal'):
    """
    Stitches two videos together.
    Intended for combining two videos that were cropped using basic_cropper (half content, half white).
    
    Args:
        video1_path (str): Path to first video (Left side or Top side).
        video2_path (str): Path to second video (Right side or Bottom side).
        output_path (str): Path to save output video.
        direction (str): 'horizontal' (Left+Right) or 'vertical' (Top+Bottom).
    """
    # Open videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened():
        print(f"Error: Could not open video {video1_path}")
        return False
    if not cap2.isOpened():
        print(f"Error: Could not open video {video2_path}")
        return False

    # Get properties from first video
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    # Verify second video properties
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if w2 != width or h2 != height:
        print(f"Warning: Resolution mismatch. Video 1: {width}x{height}, Video 2: {w2}x{h2}. Resizing Video 2 to match Video 1.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # Stop if either video ends
        if not ret1 or not ret2:
            break
            
        if w2 != width or h2 != height:
            frame2 = cv2.resize(frame2, (width, height))
            
        output_frame = np.zeros_like(frame1)
        
        if direction == 'horizontal':
            mid = width // 2
            # Take left half from frame1, right half from frame2
            output_frame[:, :mid] = frame1[:, :mid]
            output_frame[:, mid:] = frame2[:, mid:]
        elif direction == 'vertical':
            mid = height // 2
            # Take top half from frame1, bottom half from frame2
            output_frame[:mid, :] = frame1[:mid, :]
            output_frame[mid:, :] = frame2[mid:, :]
        else:
            print(f"Invalid direction: {direction}")
            cap1.release()
            cap2.release()
            out.release()
            return False
            
        out.write(output_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Stitching frame {frame_count}...", end='\r')
            
    cap1.release()
    cap2.release()
    out.release()
    print(f"\nStitching complete: {output_path}")
    return True
